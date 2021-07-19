import torch
from torch import nn, Tensor
from torch.nn import functional as F


class DINOHead(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Linear(c1, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 256)
        ])
            
        self.last_layer = nn.utils.weight_norm(nn.Linear(256, c2, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 65536):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(self.backbone.embed_dim, head_dim)

    def forward(self, x) -> Tensor:
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], dim=0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output = torch.cat((output, out))
            start_idx = end_idx

        return self.head(output)


if __name__ == '__main__':
    from xcit import XciT
    backbone = XciT('')
    model = DINO(backbone)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)