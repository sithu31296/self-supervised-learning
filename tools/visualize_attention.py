import argparse
import torch
import yaml
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from PIL import Image
from torchvision import transforms as T
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid

import sys
sys.path.insert(0, '.')
from models import get_model
from utils.utils import fix_seeds, setup_cudnn


def main(cfg):
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    device = torch.device(cfg['DEVICE'])

    # load model and weights
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['TRAIN']['IMAGE_SIZE'][0])
    model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)
    model.eval()

    response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
    img = Image.open(BytesIO(response.content)).convert('RGB')

    transform = T.Compose([
        T.Resize(cfg['TEST']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)

    # make the image divisible by patch size
    W, H = img.shape[1] - img.shape[1] % model.patch_size, img.shape[2] - img.shape[2] % model.patch_size
    img = img[:, :W, :H].unsqueeze(0)
    img = img.to(device)
    w_featmap = img.shape[-2] // model.patch_size
    h_featmap = img.shape[-1] // model.patch_size

    attentions = model(img, return_attention=True)

    # keep only the output patch attention
    attentions = attentions.squeeze()[:, 1:].view(-1, w_featmap, h_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=model.patch_size, mode='nearest')[0].detach().cpu().numpy()

    save_image(make_grid(img, normalize=True, scale_each=True), str(save_dir / "img.png"))

    for i, attn in enumerate(attentions):
        fname = save_dir / f"attn-head{i}.png"
        plt.imsave(str(fname), attn, format='png')
        print(f"{fname} saved.")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/dino.yaml', help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()
    main(cfg)