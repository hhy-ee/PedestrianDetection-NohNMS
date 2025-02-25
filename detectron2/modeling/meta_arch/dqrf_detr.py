# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from .build import META_ARCH_REGISTRY

from detectron2.structures import Boxes, ImageList, Instances
from ..postprocessing import detector_postprocess

from dqrf.backbone import build_deformable_detr_backbone
from dqrf.transformer import build_transformer
from dqrf.utils.utils import MLP, _get_clones, NestedTensor
from dqrf.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from dqrf.ch_criterion import SetCriterion as ch_criterion
from dqrf.criterion import SetCriterion as coco_criterion
from dqrf.matcher import build_matcher, build_vanilla_matcher

__all__ = ["DQRF_DETR"]

@META_ARCH_REGISTRY.register()
class DQRF_DETR(nn.Module):

    def __init__(self, cfg):
        super(DQRF_DETR, self).__init__()
        self.device = cfg.MODEL.DEVICE
        self.backbone = build_deformable_detr_backbone(cfg=cfg) # 28 870 656 parameters
        self.transformer = build_transformer(cfg) # 11 464 220

        if 'coco' in cfg.DATASETS.TRAIN[0]:
            is_coco_type_data = True
            matcher = build_vanilla_matcher(cfg)
            self.criterion = coco_criterion(cfg, matcher=matcher)
        else:
            is_coco_type_data = False
            matcher = build_matcher(cfg)
            self.criterion = ch_criterion(cfg, matcher=matcher)
        self.is_coco_type_data = is_coco_type_data
        self.aux_loss = cfg.MODEL.DQRF_DETR.AUX_LOSS
        num_queries = cfg.MODEL.DQRF_DETR.NUM_QUERIES
        num_classes = cfg.MODEL.DQRF_DETR.NUM_CLASSES
        num_layer = self.transformer.num_decoder_layers
        hidden_dim = self.transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        zero_tgt = False
        self.query_embed = nn.Embedding(num_queries, hidden_dim if zero_tgt else hidden_dim * 2)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias.data, bias_value)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = _get_clones(self.class_embed, num_layer)
        self.bbox_embed = _get_clones(self.bbox_embed, num_layer)
        # hack implementation for iterative bounding box refinement
        self.transformer.decoder.bbox_embed = self.bbox_embed

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        # self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1))
        # self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1))
        self.to(self.device)

    def forward(self, batched_inputs):

        images = self.collater(batched_inputs)

        features, pos = self.backbone(self.to_nested(images))

        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(self.input_proj(src))
            masks.append(mask)
            assert mask is not None
        query_embeds = self.query_embed.weight

        hs, memory, pos_center, dec_attns = self.transformer(srcs, masks, query_embeds, pos)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(hs)):
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])  # pred offset
            tmp += pos_center[lvl]  # add center
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'dec_attns': dec_attns, 'pos_center': pos_center[-1], 'queries': hs[-1]}
        if self.training:
            if self.aux_loss:
                output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, pos_center, hs)
            if self.is_coco_type_data:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
            else:
                targets = self.detr_mapper(batched_inputs)
            loss_dict = self.criterion(output, targets)
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_preds = output["pred_boxes"]
            orig_image_sizes = [(b.get("height"), b.get("width")) for b in batched_inputs]
            results = self.inference(box_cls, box_preds, orig_image_sizes)

            processed_results = []
            for result, _ in zip(results, orig_image_sizes):
                processed_results.append({"instances": result})
            return processed_results

    def inference(self, box_cls, box_preds, image_sizes):
        inference_dict = {
            1: self.inference_coco,
            0: self.inference_ch,
        }
        return inference_dict[self.is_coco_type_data](box_cls, box_preds, image_sizes)
    # def post_processor(self, results_per_image, height, width):
    #     post_processor_dict = {
    #         1: detector_postprocess,
    #         0: ch_detector_postprocess
    #     }
    #     return post_processor_dict[self.is_coco_type_data](results_per_image, height, width)
    def inference_coco(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        prob = box_cls.sigmoid()

        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // box_cls.shape[2]
        labels = topk_indexes % box_cls.shape[2]
        box_pred = box_cxcywh_to_xyxy(box_pred)
        box_pred = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            # result.pred_boxes = box_pred_per_image
            result.pred_boxes = Boxes(box_pred_per_image)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append({"instances": result})
            # results.append(result)
        return results

    def inference_ch(self, box_cls, box_preds, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        prob = box_cls.sigmoid()
        scores = prob.topk(k=100, dim=1).values#.squeeze(-1) # [bs, num_query, 1]
        indices = prob.topk(k=100, dim=1).indices
        labels = torch.zeros_like(scores, dtype=torch.int64, device=scores.device)#.squeeze(-1) # [bs, num_query, 1]
        box_preds = box_cxcywh_to_xyxy(box_preds)

        for i, (score, label, box_pred, indice, image_size) in enumerate(zip(scores, labels, box_preds, indices, image_sizes)):
            h, w = image_size[0], image_size[1]
            scale_fct = torch.tensor([[w, h, w, h]]).type_as(box_pred)
            box_pred = box_pred * scale_fct
            box_pred = box_pred[indice[:, 0], :]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred)
            result.scores = score[:, 0]
            result.pred_classes = label[:, 0]

        results.append(result)
        return results

    def prepare_targets(self, targets):
        """
        convert to 0, 1
        :param targets:
        :return:
        """
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes) / image_size_xyxy
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})

        return new_targets

    def collater(self, batched_inputs):
        """
        :param batched_inputs:
        :return: Structure that holds a list of images (of possibly
        varying sizes) as a single tensor.
        This works by padding the images to the same size,
        and storing in a field the original sizes of each image
        """
        # images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)

        return images
    
    def detr_mapper(self, batched_inputs):
        output_targets = []
        targets = [x["instances"] for x in batched_inputs]
        image_sizes = [x["image"].shape[1:] for x in batched_inputs]
        for bs, (target, image_size) in enumerate(zip(targets, image_sizes)):
            output_target = {}
            h, w = image_size[0], image_size[1]
            normalizer = torch.tensor([w, h, w, h], dtype=torch.float32).to(self.device)
            boxes = box_xyxy_to_cxcywh(target.get('gt_boxes').tensor.to(self.device))
            label = target.get('gt_classes').to(self.device)
            output_target['boxes'] = boxes[torch.where(label == 0)[0], :] / normalizer
            output_target['iboxes'] = boxes[torch.where(label == -1)[0], :] / normalizer
            output_target['labels'] = label.new_full((len(torch.where(label == 0)[0]),), 1)
            output_target['size'] = torch.tensor([h, w], dtype=torch.int64).to(self.device)
            output_targets.append(output_target)
        return output_targets

    def to_nested(self, images):
        #ok
        b, c, h, w = images.tensor.size()
        tensor = torch.zeros_like(images.tensor, device=self.device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=self.device)
        for img_size, img, pad_img, m in zip(images.image_sizes, images.tensor, tensor, mask):
            pad_img[:, :img_size[0], :img_size[1]].copy_(img[:, :img_size[0], :img_size[1]])
            m[:img_size[0], :img_size[1]] = False
        return NestedTensor(tensor, mask)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, pos_center, queries):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b, 'pos_center': c, 'queries': d} for a, b, c, d in
                zip(outputs_class[:-1], outputs_coord[:-1], pos_center[:-1], queries[:-1])]

def ch_detector_postprocess(results: Instances, output_height: int, output_width: int):
    """ Same as detectron2's detector_postprocess except that we don't clip boxes outside range
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    # output_boxes.clip(results.image_size) removing box clipping

    results = results[output_boxes.nonempty()]

    return results