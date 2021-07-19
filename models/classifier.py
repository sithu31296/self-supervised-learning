import torch
from torch import nn, Tensor


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1))


if __name__ == '__main__':
    model = LinearClassifier(384)
    x = torch.randn(1, 384)
    y = model(x)
    print(y.shape)