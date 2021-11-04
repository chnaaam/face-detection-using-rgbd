import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        print(X.shape)
        X = self.conv1(X)

        print(X.shape)

if __name__ == "__main__":
    a = torch.ones(size=(1, 3, 68, 69))

    model = TestModel()

    model(a)