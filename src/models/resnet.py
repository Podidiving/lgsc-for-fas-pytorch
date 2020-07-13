import torch
from torch import nn
from torchvision import models

from typing import List


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_layer(
    block, inplanes, planes, blocks, stride=1, dilation=1, norm_layer=None
):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            groups=1,
            base_width=64,
            dilation=dilation,
            norm_layer=norm_layer,
        )
    )
    inplanes = planes * block.expansion

    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=1,
                base_width=64,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(*layers)


class ResNet18Encoder(nn.Module):
    def __init__(
        self, out_indices: List[int] = (1, 2, 3, 4), pretrained: bool = True
    ):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = None
        self.out_indices = out_indices

        self._freeze_encoder()

    def _freeze_encoder(self):
        for p in self.resnet18.parameters():
            p.requires_grad = False

    def forward(self, input):
        outs = []
        x = self.resnet18.conv1(input)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        outs.append(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        if 1 in self.out_indices:
            outs.append(x)
        x = self.resnet18.layer2(x)
        if 2 in self.out_indices:
            outs.append(x)
        x = self.resnet18.layer3(x)
        if 3 in self.out_indices:
            outs.append(x)
        x = self.resnet18.layer4(x)
        if 4 in self.out_indices:
            outs.append(x)
        return outs


class ResNet18Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(
            in_features=self.resnet18.fc.in_features, out_features=num_classes
        )
        self.drop = nn.Dropout(dropout)
        self._freeze_clf()

    def _freeze_clf(self):
        for p in self.resnet18.parameters():
            p.requires_grad = False
        for p in self.resnet18.fc.parameters():
            p.requires_grad = True

    def forward(self, input):
        x = self.resnet18.conv1(input)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.resnet18.fc(x)
        return x
