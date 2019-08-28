# Res2Net for Instance segmentation and Object detection using MaskRCNN

## Introduction
This repo uses *MaskRCNN* as the baseline method for Instance segmentation and Object detection. We use the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) as the baseline. (TODO: merge this repo into the maskrcnn-benchmark.)

[Res2Net](https://github.com/gasvn/Res2Net) is a powerful backbone architecture that can be easily implemented into state-of-the-art models by replacing the bottleneck with Res2Net module.
More detail can be found on [ "Res2Net: A New Multi-scale Backbone Architecture"](https://arxiv.org/pdf/1904.01169.pdf) and our [project page](https://mmcheng.net/res2net) .


## Performance

### Results on Instance segmentation and Object detection using MaskRCNN.

**Performance on Instance segmentation:**

| Backbone     | Setting      | AP      | AP50    | AP75     | APs       |APm        |    APl    |
|--------------|--------------|---------|---------|----------|-----------|-----------|-----------|
| ResNet-50    |  64w         | 33.9    | 55.2    | 36.0     | 14.8      | 36.0      | 50.9      |
| ResNet-50    |  48w×2s      | 34.2    | 55.6    | 36.3     | 14.9      | 36.8      | 50.9      |
| Res2Net-50   |  26w×4s      | 35.6    | 57.6    | 37.6     | 15.7      | 37.9      | 53.7      |
| Res2Net-50   |  18w×6s      | 35.7    | 57.5    | 38.1     | 15.4      | 38.1      | 53.7      |
| Res2Net-50   |  14w×8s      | 35.3    | 57.0    | 37.5     | 15.6      | 37.5      | 53.4      |
| ResNet-101   |  64w         | 35.5    | 57.0    | 37.9     | 16.0      | 38.2      | 52.9      |
| Res2Net-101  |  26w×4s      | 37.1    | 59.4    | 39.4     | 16.6      | 40.0      | 55.6      |

**Performance on Object detection:**

| Backbone     | Setting      | AP      | AP50    | AP75     | APs       |APm        |    APl    |
|--------------|--------------|---------|---------|----------|-----------|-----------|-----------|
| ResNet-50    |  64w         | 37.5    | 58.4    | 40.3     | 20.6      | 40.1      | 49.7      |
| ResNet-50    |  48w×2s      | 38.0    | 58.9    | 41.3     | 20.5      | 41.0      | 49.9      |
| Res2Net-50   |  26w×4s      | 39.6    | 60.9    | 43.1     | 22.0      | 42.3      | 52.8      |
| Res2Net-50   |  18w×6s      | 39.9    | 60.9    | 43.3     | 21.8      | 42.8      | 53.7      |
| Res2Net-50   |  14w×8s      | 39.1    | 60.2    | 42.1     | 21.7      | 41.7      | 52.8      |
| ResNet-101   |  64w         | 39.6    | 60.6    | 43.2     | 22.0      | 43.2      | 52.4      |
| Res2Net-101  |  26w×4s      | 41.8    | 62.6    | 45.6     | 23.4      | 45.5      | 55.6      |


(Noted that pretrained models trained with pytorch usually achieve slightly worse performance than the caffe pretrained models, we took [advice](https://github.com/facebookresearch/maskrcnn-benchmark/issues/504) from the author of MaskRCNN-benchmark to use 2x schedule in all experiments including baseline and our method.)

## Applications
Other applications such as Classification,  Semantic segmentation, pose estimation, Class activation map can be found on https://mmcheng.net/res2net/ and https://github.com/gasvn/Res2Net .

## Installation
(**This repo is based on the [mask-rcnn benchmark]((https://github.com/facebookresearch/maskrcnn-benchmark))**, the useage is remain the same with the original repo.)

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Perform training on COCO dataset

For the following examples to work, you need to first install `maskrcnn_benchmark`.

You will also need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations)

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
# or use COCO 2017 version
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

# for pascal voc dataset:
ln -s /path_to_VOCdevkit_dir datasets/voc
```

P.S. `COCO_2017_train` = `COCO_2014_train` + `valminusminival` , `COCO_2017_val` = `minival`
      

You can also configure your own paths to the datasets.
For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to
point to the location where your dataset is stored.
You can also create a new `paths_catalog.py` file which implements the same two classes,
and pass it as a config argument `PATHS_CATALOG` during training.

### Single GPU training

Most of the configuration files that we provide assume that we are running on 8 GPUs.
In order to be able to run it on fewer GPUs, there are a few possibilities:

**1. Run the following without modifications**

```bash
python /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the
configuration files a global batch size that is divided over the number of GPUs. So if we only
have a single GPU, this means that the batch size for that GPU will be 8x larger, which might lead
to out-of-memory errors.

If you have a lot of memory available, this is the easiest solution.

**2. Modify the cfg parameters**

If you experience out-of-memory errors, you can reduce the global batch size. But this means that
you'll also need to change the learning rate, the number of iterations and the learning rate schedule.

Here is an example for Mask R-CNN Res2Net-50 FPN with the 2x schedule:
```bash
python tools/train_net.py --config-file "configs/pytorch_mask_rcnn_R2_50_s4_FPN_2x.ymal" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1
```
This follows the [scheduling rules from Detectron.](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30)
Note that we have multiplied the number of iterations by 8x (as well as the learning rate schedules),
and we have divided the learning rate by 8x.

We also changed the batch size during testing, but that is generally not necessary because testing
requires much less memory than training.


### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "configs/pytorch_mask_rcnn_R2_50_s4_FPN_2x.ymal"
```


## Inference in a few lines
We provide a helper class to simplify writing inference pipelines using pre-trained models.
Here is how we would do it. Run this from the `demo` folder:
```python
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/pytorch_mask_rcnn_R2_50_s4_FPN_2x.ymal"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = ...
predictions = coco_demo.run_on_opencv_image(image)
```

## Adding your own dataset

This implementation adds support for COCO-style datasets.
But adding support for training on a new dataset can be done as follows:
```python
from maskrcnn_benchmark.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, ...):
        # as you would do normally

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}
```
That's it. You can also add extra fields to the boxlist, such as segmentation masks
(using `structures.segmentation_mask.SegmentationMask`), or even your own instance type.

For a full example of how the `COCODataset` is implemented, check [`maskrcnn_benchmark/data/datasets/coco.py`](maskrcnn_benchmark/data/datasets/coco.py).




## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2019}
}
@misc{massa2018mrcnn,
author = {Massa, Francisco and Girshick, Ross},
title = {{maskrnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch}},
year = {2018},
howpublished = {\url{https://github.com/facebookresearch/maskrcnn-benchmark}},
note = {Accessed: [Insert date here]}
}
```
## Acknowledge
This code is partly borrowed from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.
