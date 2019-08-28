# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the Res2Net module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = Res2Net(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "Res2Net50StagesTo4",
    )
OR:
    model = Res2Net(
        "StemWithGN",
        "BottleneckWithGN",
        "Res2Net50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry
import math

# Res2Net stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard Res2Net models
# -----------------------------------------------------------------------------
# Res2Net-50 (including all stages)
Res2Net50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# Res2Net-50 up to stage 4 (excludes stage 5)
Res2Net50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# Res2Net-101 (including all stages)
Res2Net101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# Res2Net-101 up to stage 4 (excludes stage 5)
Res2Net101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# Res2Net-50-FPN (including all stages)
Res2Net50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# Res2Net-101-FPN (including all stages)
Res2Net101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# Res2Net-152-FPN (including all stages)
Res2Net152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class Res2Net(nn.Module):
    def __init__(self, cfg):
        super(Res2Net, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified Res2Net stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        scale = cfg.MODEL.RESNETS.SCALE
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                scale=scale
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class Res2NetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        scale = 4,
    ):
        super(Res2NetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                scale=scale
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    scale=4
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                scale = scale
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottle2neck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        scale = 4
    ):
        super(Bottle2neck, self).__init__()

        self.downsample = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
            self.stype = 'stage'
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride_3x3, padding=dilation)
        else:
            self.stype = 'normal'
        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels*scale,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(
              bottleneck_channels, 
              bottleneck_channels, 
              kernel_size=3, 
              stride = stride_3x3, 
              padding=dilation, 
              groups=num_groups,
              dilation=dilation,
              bias=False
              ))
          bns.append(norm_func(bottleneck_channels))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = Conv2d(
            bottleneck_channels*scale, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        self.scale = scale
        self.width  = bottleneck_channels

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)
        for l in self.convs:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = F.relu_(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out

class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class Bottle2neckWithFixedBatchNorm(Bottle2neck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        scale=4
    ):
        super(Bottle2neckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            scale=scale,
            norm_func=FrozenBatchNorm2d
        )

class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class Bottle2neckWithGN(Bottle2neck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        scale=4
    ):
        super(Bottle2neckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            scale=4,
            norm_func=group_norm
        )

class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "Bottle2neckWithFixedBatchNorm": Bottle2neckWithFixedBatchNorm,
    "Bottle2neckWithGN": Bottle2neckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R2-50-C4": Res2Net50StagesTo4,
    "R2-50-C5": Res2Net50StagesTo5,
    "R2-101-C4": Res2Net101StagesTo4,
    "R2-101-C5": Res2Net101StagesTo5,
    "R2-50-FPN": Res2Net50FPNStagesTo5,
    "R2-50-FPN-RETINANET": Res2Net50FPNStagesTo5,
    "R2-101-FPN": Res2Net101FPNStagesTo5,
    "R2-101-FPN-RETINANET": Res2Net101FPNStagesTo5,
    "R2-152-FPN": Res2Net152FPNStagesTo5,
})
