import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbone_layer import BackboneLayer
from nets.depth_backbone_layer import DepthBackboneLayer

class FPN(nn.Module):
    def __init__(self, with_depth_info=True):
        super(FPN, self).__init__()

        # Bottom-up Layers
        self.backbone_layer = BackboneLayer()

        self.with_depth_info = with_depth_info

        in_channel_list = [2048, 1024, 512, 256]

        if self.with_depth_info:
            self.depth_backbone_layer = DepthBackboneLayer()

            in_channel_list = [in_channel * 2 for in_channel in in_channel_list]

        # Top-dowm layers
        # upsample C5 to get P5 from the FPN paper
        self.p5_lateral_layer = nn.Sequential(
            nn.Conv2d(in_channel_list[0], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p4_lateral_layer = nn.Sequential(
            nn.Conv2d(in_channel_list[1], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p3_lateral_layer = nn.Sequential(
            nn.Conv2d(in_channel_list[2], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p2_lateral_layer = nn.Sequential(
            nn.Conv2d(in_channel_list[3], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # self.p5_upsampling_layer = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p4_upsampling_layer = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p3_upsampling_layer = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p2_upsampling_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.p6_conv_layer = nn.Sequential(
            nn.Conv2d(in_channel_list[3] if self.with_depth_info else 256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p5_conv_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p4_conv_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.p3_conv_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, a=1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, rgb_imgs, depth_imgs=None):
        c2, c3, c4, c5, c6 = self.backbone_layer(rgb_imgs)
        # c3, c4, c5 = self.backbone_layer(X)

        if self.with_depth_info:
            d2, d3, d4, d5, d6 = self.depth_backbone_layer(depth_imgs)

            # gamma = 0.7
            #
            # c2 = c2 * (d2 * gamma + gamma * (1 - d2))
            # c3 = c3 * (d3 * gamma + gamma * (1 - d3))
            # c4 = c4 * (d4 * gamma + gamma * (1 - d4))
            # c5 = c5 * (d5 * gamma + gamma * (1 - d5))
            # c6 = c6 * (d6 * gamma + gamma * (1 - d6))

            c2 = torch.cat([c2, d2], dim=1)
            c3 = torch.cat([c3, d3], dim=1)
            c4 = torch.cat([c4, d4], dim=1)
            c5 = torch.cat([c5, d5], dim=1)
            c6 = torch.cat([c6, d6], dim=1)



        p5_lateral = self.p5_lateral_layer(c5) # p5_lateral: (256, 20, 20)
        p4_lateral = self.p4_lateral_layer(c4) # p4_lateral: (256, 40, 40)
        p3_lateral = self.p3_lateral_layer(c3) # p3_lateral: (256, 80, 80)
        p2_lateral = self.p2_lateral_layer(c2) # p2_lateral: (256, 160, 160)

        p6 = self.p6_conv_layer(c6)
        p5 = p5_lateral

        upsampled_p5 = F.interpolate(p5_lateral, size=p4_lateral.shape[-2:], mode="nearest")
        p4 = p4_lateral + upsampled_p5
        p4 = self.p4_conv_layer(p4)

        upsampled_p4 = F.interpolate(p4_lateral, size=p3_lateral.shape[-2:], mode="nearest")
        p3 = p3_lateral + upsampled_p4
        p3 = self.p3_conv_layer(p3)

        upsampled_p3 = F.interpolate(p3_lateral, size=p2_lateral.shape[-2:], mode="nearest")
        p2 = p2_lateral + upsampled_p3
        p2 = self.p3_conv_layer(p2)

        return p2, p3, p4, p5, p6
        # return p3, p4, p5

if __name__ == "__main__":
    rgb_sample = torch.ones(size=(1, 3, 640, 640))
    depth_sample = torch.ones(size=(1, 3, 640, 640))

    fpn = FPN(with_depth_info=True)

    fpn(rgb_sample, depth_sample)