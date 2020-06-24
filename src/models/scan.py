from torch import nn

from models.decoder import Decoder
from models.resnet import ResNet18Classifier, ResNet18Encoder


class SCAN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = ResNet18Encoder()
        self.decoder = Decoder()
        self.clf = ResNet18Classifier(dropout=dropout)

    def forward(self, x):
        outs = self.backbone(x)
        outs = self.decoder(outs)

        s = x + outs[-1]
        clf_out = self.clf(s)

        return outs, clf_out
