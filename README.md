# Pedestrain Detection baseline

The Pytoch implementation for the Pedestrian detection baseline.

* This repo is implemented based on [detectron2](https://github.com/facebookresearch/detectron2).

## Performance
|    Model    | Backbone |  AP  |  Recall |  MR  | Weights |
|-------------|----------|------|---------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Faster RCNN(anchor=1) | ResNet-50| 85.50 |   88.39  | 44.82 |         |
| Faster RCNN(anchor=3) | ResNet-50| 85.47 |   88.20  | 44.29 |         |
|  Retinanet(anchor=9)  | ResNet-50| 76.59 |   83.06  | 62.76 |         |

## Preparation
Download the CrowdHuman Datasets from http://www.crowdhuman.org/, and then move them under the directory like:
```
./datasets/crowdhuman
├── annotations
│   └── annotation_train.odgt
│   └── annotation_val.odgt
├── images
│   └── train
│   └── val
```

Download the pretrained model, and then move them under the directory like:
```
./detectron2/ImageNetPretrained/MSRA/R-50.pkl
```

## Installation
```
  conda create -n ped python=3.7 -y
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
  cd root_dir
  pip install -e . 
  cd dqrf/ops
  pip install -e .
  pip install pycocotools, scipy, opencv-python, pandas
```

## Training in Command Line
Train Faster RCNN on 8 gpus:
```
python tools/train_net.py --num-gpus 8 --config-file configs/CrowdHuman/faster_rcnn_R_50_FPN_baseline_iou_0.5.yaml
```

## Quick Start
See [GETTING_STARTED.md](GETTING_STARTED.md) in detectron2

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

## Citation
if you find this project useful for your research, please cite:
```
Please waiting!
```
