# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    CrowdHumanEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetCatalog,
    MetadataCatalog)

from dqrf import add_dqrf_config, add_dataset_path
from dqrf.utils.dataset_mapper import DqrfDatasetMapper, CH_DqrfDatasetMapper
from dqrf.utils.get_crowdhuman_dicts import get_crowdhuman_dicts

logger = logging.getLogger("detectron2")

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_norm_val = 0.0
        super().__init__(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of parameters {n_parameters}")

        def is_backbone(n, backbone_names):
            out = False
            for b in backbone_names:
                if b in n:
                    out = True
                    break
            return out

        #careful DEFORMABLE DETR yields poorer performance is its FAKE FPN is trained on the same LR as Resnet
        #Resnet parameters name is backbone.0
        lr_backbone_names = ['backbone.0']

        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not is_backbone(n, lr_backbone_names) and
                           not (
                                       "roi_fc1" in n or "roi_fc2" in n or "offset" in n or "sampling_locs" in n or "sampling_cens" in n or "sampling_weight" in n or "conv_offset" in n or 'learnable_fc' in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR,
            },
            {
                "params": [p for n, p in model.named_parameters() if is_backbone(n, lr_backbone_names) and
                           not (
                                       "roi_fc1" in n or "roi_fc2" in n or "offset" in n or "sampling_locs" in n or "sampling_cens" in n or "sampling_weight" in n or "conv_offset" in n or 'learnable_fc' in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_locs" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.SAMPLE_MULTIPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_cens" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.CENTER_MULTPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_weight" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.WEIGHT_MULTIPLIER,
            },

        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = CH_DqrfDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "crowdhuman":
            return CrowdHumanEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dqrf_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # crowdhuman dataset for detr
    add_dataset_path(cfg)
    ch_train = get_crowdhuman_dicts(cfg.CH_PATH.ANNOT_PATH_TRAIN, cfg.CH_PATH.IMG_PATH_TRAIN)
    ch_val = get_crowdhuman_dicts(cfg.CH_PATH.ANNOT_PATH_VAL, cfg.CH_PATH.IMG_PATH_VAL)
    DatasetCatalog.register(cfg.DATASETS.TRAIN[0], ch_train)
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["background", "person"])
    DatasetCatalog.register(cfg.DATASETS.TEST[0], ch_val)
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=["background", "person"])
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(json_file=cfg.CH_PATH.ANNOT_PATH_VAL)
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(gt_dir=cfg.CH_PATH.IMG_PATH_VAL)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.dist_url = 'tcp://127.0.0.1:50152'
    # args.num_gpus = 2
    # args.config_file = 'configs/CrowdHuman/faster_rcnn_R_50_FPN_baseline_anchor_9_iou_0.5.yaml'
    # args.config_file = 'configs/CrowdHuman/dqrf_detr_baseline.yaml'
    # args.config_file = 'configs/CrowdHuman/retinanet_R_50_FPN_baseline_iou_0.5.yaml'

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
