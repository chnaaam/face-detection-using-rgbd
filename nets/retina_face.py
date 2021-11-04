import torch
import torch.nn as nn

from nets.feature_pyramid_network import FPN
from nets.context_module import ContextModule
from nets.heads import ClassificationHead, RegressionHead
from regression import RegressionTransform
from prior_box import Anchors
from multi_task_loss import MultiTaskLoss

class RetinaFace(nn.Module):
    def __init__(self, with_depth_info=True):
        super(RetinaFace, self).__init__()

        # Components
        # 1. Feature pyramid network
        #    - The FPN gets the input face images and outputs five feature maps of different scale
        # 2. Context module
        #    - The Context module gets a feature map of a particular scale
        # 3. Multi-task loss

        self.with_depth_info = with_depth_info
        self.fpn = FPN(with_depth_info=with_depth_info)
        self.anchors = Anchors()
        self.context_module_list = nn.ModuleList(modules=[ContextModule(in_channels=256) for _ in range(5)])
        self.classification_heads = nn.ModuleList(modules=[ClassificationHead(in_channels=256, n_anchors=3) for _ in range(5)])
        self.regression_heads = nn.ModuleList(modules=[RegressionHead(in_channels=256, n_anchors=3) for _ in range(5)])
        self.regression = RegressionTransform()
        self.criterion = MultiTaskLoss()
        self.freeze_bn()
        self.freeze_pretrained_model()

    def forward(self, rgb_imgs, depth_imgs=None, targets=None):
        # X: image (batch size, image channels, image height, image width)
        fpn_outs = self.fpn(rgb_imgs, depth_imgs)

        cm_outs = [self.context_module_list[i](fpn_out) for i, fpn_out in enumerate(fpn_outs)]
        cls_outs = [self.classification_heads[i](cm_out) for i, cm_out in enumerate(cm_outs)]
        reg_outs = [self.regression_heads[i](cm_out) for i, cm_out in enumerate(cm_outs)]

        cls_outs = torch.cat(cls_outs, dim=1)
        reg_outs = torch.cat(reg_outs, dim=1)

        if targets is not None:
            return self.criterion(cls_outs, reg_outs, self.anchors.forward(rgb_imgs), targets)

        else:
            bboxes = self.regression(self.anchors.forward(rgb_imgs), reg_outs, rgb_imgs)

            return cls_outs, bboxes

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_pretrained_model(self):
        for name, param in self.named_parameters():
            if self.with_depth_info:
                if "fpn.backbone_layer.body" in name or "fpn.depth_backbone_layer.body" in name:
                    param.requires_grad = False
            else:
                if "fpn.backbone_layer.body" in name:
                    param.requires_grad = False


if __name__ == "__main__":
    sample = torch.ones(size=(1, 3, 640, 640))
    sample1 = torch.ones(size=(1, 3, 640, 640))
    model = RetinaFace(with_depth_info=True)
    model(sample, sample1)