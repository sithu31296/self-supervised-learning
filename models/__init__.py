from .vit import ViT
from .xcit import XciT
from .dino import DINO

methods = {
    'dino': DINO
}

__all__ = {
    'vit': ViT,
    'xcit': XciT
}

def get_model(model: str, variant: str, img_size):
    assert model in __all__.keys()
    return __all__[model](variant, image_size=img_size)

def get_method(method: str, model: str, variant: str, img_size, head_dim):
    assert method in __all__.keys()
    backbone = get_model(model, variant, img_size)
    return methods[method](backbone, head_dim)