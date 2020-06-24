import torch
from torch import nn
from torch.nn import functional as F

from models.resnet import make_layer, BasicBlock


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=(64, 64, 128, 256, 512),
        out_channels=(512, 256, 128, 64, 64, 3),
        num_outs=6,
    ):
        super().__init__()
        assert isinstance(in_channels, tuple)
        self.in_channels = in_channels  # [64, 64, 128, 256, 512]
        self.out_channels = out_channels  # [512, 256, 128, 64, 64, 3]
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.deres_layers = []
        self.conv2x2 = []
        self.conv1x1 = []
        for i in range(self.num_ins - 1, -1, -1):  # 43210
            deres_layer = make_layer(
                BasicBlock,
                inplanes=128 if i == 1 else in_channels[i],
                planes=out_channels[-i - 1],
                blocks=2,
                stride=1,
                dilation=1,
                norm_layer=nn.InstanceNorm2d,
            )
            out2x2 = in_channels[i] if i < 2 else int(in_channels[i] / 2)
            conv2x2 = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels=out2x2, kernel_size=2),
                nn.InstanceNorm2d(out2x2),
                nn.ReLU(),
            )
            conv1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=128 if i == 1 else in_channels[i],
                    out_channels=out_channels[-i - 1],
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(out_channels[-i - 1]),
            )
            self.deres_layers.append(deres_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)

        self.deres_layers: nn.ModuleList = nn.ModuleList(self.deres_layers)
        self.conv2x2: nn.ModuleList = nn.ModuleList(self.conv2x2)
        self.conv1x1: nn.ModuleList = nn.ModuleList(self.conv1x1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        out = inputs[-1]
        outs.append(out)

        for i in range(self.num_ins):
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            out = F.pad(out, [0, 1, 0, 1])
            out = self.conv2x2[i](out)
            if i < 4:
                out = torch.cat([out, inputs[-i - 2]], dim=1)
            identity = self.conv1x1[i](out)
            out = self.deres_layers[i](out) + identity
            outs.append(out)
        outs[-1] = torch.tanh(outs[-1])

        return outs
