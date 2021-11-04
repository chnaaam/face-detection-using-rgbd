import torch
import torch.nn as nn

import torchvision
from torchvision.models import _utils

class BackboneLayer(nn.Module):
    def __init__(
            self,
            out_channels=256,
            return_layers={
                "layer1": 0,
                "layer2": 1,
                "layer3": 2,
                "layer4": 3},
            pretained_model="resnet50",
            pretrained=True):

        super(BackboneLayer, self).__init__()

        if pretained_model == "resnet50":
            resnet50 = torchvision.models.resnet50(pretrained=pretrained)
            self.body = _utils.IntermediateLayerGetter(resnet50, return_layers)
        else:
            raise NotImplementedError()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, X):
        out1 = self.body(X)
        out2 = self.layer2(out1[3])

        return out1[0], out1[1], out1[2], out1[3], out2
        # return out1[1], out1[2], out1[3]

if __name__ == "__main__":
    sample = torch.ones(size=(1, 3, 640, 640))
    backbone_layer = BackboneLayer(out_channels=512)

    backbone_layer(sample)