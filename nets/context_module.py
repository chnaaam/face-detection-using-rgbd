import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True))

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, X):
        # X: (256, A, A)
        # X1: (256, A, A)
        # X2: (128, A, A)
        # X3: (64, A, A)
        # X4: (64, A, A)

        X1 = self.layer1(X)

        X2 = self.layer2(X1)
        X3 = self.layer3(X2)
        X4 = self.layer4(X3)

        X_out = torch.cat([X2, X3, X4], dim=1)
        X_out = self.layer5(X_out)
        # X_out = F.relu(X_out)
        return X_out

if __name__ == "__main__":
    sample = torch.ones(size=(1, 256, 160, 160))
    context_module = ContextModule(in_channels=256)

    context_module(sample)