import argparse
import torch
from torch.nn.modules import linear
import yaml
import time
from tqdm import tqdm
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '.')
from datasets.imagenet import ImageNet
from datasets.transforms import DINOAug
from models import get_model, get_backbone
from models.classifier import LinearClassifier
from utils.utils import fix_seeds, time_synchronized, setup_cudnn, setup_ddp
from utils import get_scheduler
from utils.metrics import accuracy


def main(cfg):
    start = time_synchronized()
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    device = torch.device(cfg['DEVICE'])
    ddp_enable = cfg['TRAIN']['DDP']['ENABLE']
    epochs = cfg['TRAIN']['EPOCHS']
    gpu = setup_ddp()

    # setup augmentation, dataset and dataloader
    train_transform = T.Compose([
        T.RandomResizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainset = ImageNet(cfg['DATASET']['ROOT'], split='train', transform=train_transform)
    valset = ImageNet(cfg['DATASET']['ROOT'], split='val', transform=val_transform)
    sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)
    
    # load model and classifier
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['TRAIN']['IMAGE_SIZE'][0])
    model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    linear_classifier = LinearClassifier(model.embed_dim, cfg['EVAL']['NUM_CLASSES'])
    linear_classifier = linear_classifier.to(device)
    
    if ddp_enable:
        linear_classifier = DDP(linear_classifier, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(linear_classifier.parameters(), lr=cfg['TRAIN']['LR'], momentum=0.9, weight_decay=0)
    scheduler = get_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    writer = SummaryWriter(save_dir / 'logs')

    iters_per_epoch = int(len(trainset)) / cfg['TRAIN']['BATCH_SIZE']

    for epoch in range(1, epochs+1):
        linear_classifier.train()
        
        if ddp_enable:
            trainloader.sampler.set_epoch(epoch)

        train_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")
        
        for iter, (img, target) in pbar:
            img = img.to(device)
            target = target.to(device)

            with torch.no_grad():
                pred = model(img)
            
            with autocast(enabled=cfg['TRAIN']['AMP']):
                pred = linear_classifier(pred)
                loss = loss_fn(pred, target)

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() * img.shape[0]

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= len(trainset)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()

        scheduler.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if epoch > cfg['TRAIN']['EVAL_INTERVAL'] and epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0:
            linear_classifier.eval()
            val_loss = 0.0
            for img, target in valloader:
                img = img.to(device)
                target = target.to(device)

                with torch.no_grad():
                    pred = model(img)

                pred = linear_classifier(pred)
                loss = loss_fn(pred, target)

                acc1, acc5 = accuracy(pred, target, topk=(1, 5))

                val_loss += loss.item() * img.shape[0]

            val_loss /= len(valset)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)


    writer.close()
    pbar.close()

    end = time.gmtime(time_synchronized() - start)
    total_time = time.strftime("%H:%M:%S", end)

    print(f"Total Training Time: {total_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()
    main(cfg)