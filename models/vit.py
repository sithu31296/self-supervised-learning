import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor



class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
    
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'

        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.flatten(2).swapaxes(1, 2)     # b x (14*14) x hidden_dim

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=12):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x += self.drop_path(y)
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


vit_settings = {
    'T': [8, 12, 192, 3, 0.1],     #[patch_size, number_of_layers, embed_dim, heads]
    'S': [8, 12, 384, 6, 0.1],
    'B': [8, 12, 768, 12, 0.1]
}


class ViT(nn.Module):
    def __init__(self, model_name: str = 'S', pretrained: str = None, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in vit_settings.keys(), f"DeiT model name should be in {list(vit_settings.keys())}"
        patch_size, layers, embed_dim, heads, drop_path_rate = vit_settings[model_name]

        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, heads, dpr[i])
        for i in range(layers)])

        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x: Tensor, W: int, H: int) -> Tensor:
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patches == N and H == W:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = W // self.patch_size
        h0 = H // self.patch_size

        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic'
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        B, C, W, H = x.shape
        x = self.patch_embed(x)             
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.interpolate_pos_encoding(x, W, H)
        
        for i, blk in enumerate(self.blocks):
            if i + 1 == len(self.blocks):
                if return_attention:
                    return blk(x, return_attention=return_attention)[:, :, 0, :]
            x = blk(x)

        return self.norm(x[:, 0])
        

if __name__ == '__main__':
    model = ViT('B')
    model.load_state_dict(torch.load('checkpoints/vit/dino_vitbase8_pretrain.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 224, 224)
    print(model(x, return_attention=True).shape)