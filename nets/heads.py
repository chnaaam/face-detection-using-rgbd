import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, n_anchors=3):
        super(ClassificationHead, self).__init__()
        self.n_anchors = n_anchors
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=n_anchors * 2, kernel_size=1, stride=1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        out = self.layer(X)
        # out = out.permute(0, 2, 3, 1).contiguous()
        # out = out.view(out.shape[0], -1, 2)
        # out = self.softmax(out)
        # return out

        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.n_anchors, 2)
        out = self.softmax(out)

        return out.contiguous().view(out.shape[0], -1, 2)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, n_anchors=3):
        super(RegressionHead, self).__init__()

        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=n_anchors * 4, kernel_size=1, stride=1)

    def forward(self, X):
        out = self.layer(X)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 4)

        return out

if __name__ == "__main__":
    sample = torch.ones(size=(1, 256, 160, 160))
    cls_head = ClassificationHead(in_channels=256)
    cls_head(sample)

    reg_head = RegressionHead(in_channels=256)
    reg_head(sample)