import argparse
import torch
import yaml
from torch.nn import functional as F
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torch import distributed as dist

import sys
sys.path.insert(0, '.')
from datasets.imagenet import ImageNet
from models import get_model
from utils.utils import fix_seeds, setup_cudnn, setup_ddp


def main(cfg):
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    device = torch.device(cfg['DEVICE'])
    _ = setup_ddp()

    # setup augmentation, dataset and dataloader
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ReturnIndexDataset(cfg['DATASET']['ROOT'], split='train', transform=transform)
    valset = ReturnIndexDataset(cfg['DATASET']['ROOT'], split='val', transform=transform)
    sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)
    
    # student and teacher networks
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['EVAL']['IMAGE_SIZE'][0])
    model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)
    model.eval()

    # extract features
    train_features = extract_features(model, trainloader, device)
    val_features = extract_features(model, valloader, device)

    train_features = F.normalize(train_features, p=2, dim=1)
    val_features = F.normalize(val_features, p=2, dim=1)

    train_labels = torch.tensor([s[-1] for s in trainset.samples]).long()
    val_labels = torch.tensor([s[-1] for s in valset.samples]).long()

    for k in cfg['EVAL']['KNN']['NB_KNN']:
        top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, k, cfg['EVAL']['KNN']['TEMP'], cfg['EVAL']['NUM_CLASSES'], device)
        print(f"{k}-NN classifier results >> Top1: {top1}, Top5: {top5}")


class ReturnIndexDataset(ImageNet):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, idx


@torch.no_grad()
def extract_features(model, dataloader, device):
    features = None
    for img, index in dataloader:
        img = img.to(device)
        index = index.to(device)

        feats = model(img).clone()

        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(dataloader.dataset), feats.shape[-1])
            features = features.to(device)

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = dist.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()

        # share features between processes
        feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1), dtype=feats.dtype, device=device)
        output_l = list(feats_all.unbind(0))
        output_all_reduce = dist.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        if dist.get_rank() == 0:
            features.index_copy_(0, torch.cat(y_l), torch.cat(output_l))

    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, temp, num_classes, device):
    top1, top5, total = 0.0, 0.0, 0

    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks

    retrieval_one_hot = torch.zeros(k, num_classes, device=device)

    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx:min((idx+imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx:min((idx+imgs_per_chunk), num_test_images)]
        
        # calculate dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k)
        candidates = train_labels.view(1, -1).expand(targets.shape[0], -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(targets.shape[0] * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temp).exp_()

        probs = torch.sum(torch.mul(
            retrieval_one_hot.view(targets.shape[0], -1, num_classes),
            distances_transform.view(targets.shape[0], -1, 1)
        ), dim=1)

        _, preds = probs.sort(1, descending=True)

        # find the preds that match the target
        correct = preds.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        top5 += correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)

    top1 *= 100.0 / total
    top5 *= 100.0 / total

    return top1, top5
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()
    main(cfg)