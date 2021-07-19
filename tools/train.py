import argparse
import torch
import yaml
import time
from tqdm import tqdm
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '.')
from datasets.imagenet import ImageNet
from datasets.transforms import DINOAug
from models import get_method
from utils.utils import fix_seeds, time_synchronized, setup_cudnn, setup_ddp
from utils import get_scheduler
from utils.loss import DINOLoss


def main(cfg):
    start = time_synchronized()
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    device = torch.device(cfg['DEVICE'])
    ddp_enable = cfg['TRAIN']['DDP']
    epochs = cfg['TRAIN']['EPOCHS']

    gpu = setup_ddp()

    # setup augmentation, dataset and dataloader
    transform = DINOAug(cfg['TRAIN']['IMAGE_SIZE'], cfg['TRAIN']['DINO']['CROP_SCALE'], cfg['TRAIN']['DINO']['LOCAL_CROPS'])
    dataset = ImageNet(cfg['DATASET']['ROOT'], split='train', transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=sampler)
    
    # student and teacher networks
    student = get_method(cfg['METHOD'], cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['TRAIN']['IMAGE_SIZE'][0], cfg['TRAIN']['DINO']['HEAD_DIM'])
    teacher = get_method(cfg['METHOD'], cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['TRAIN']['IMAGE_SIZE'][0], cfg['TRAIN']['DINO']['HEAD_DIM'])
    student, teacher = student.to(device), teacher.to(device)

    if ddp_enable:
        student = DDP(student, device_ids=[gpu])
        teacher.load_state_dict(student.module.state_dict())
    else:
        teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = DINOLoss(cfg['TRAIN']['DINO']['HEAD_DIM'], cfg['TRAIN']['DINO']['LOCAL_CROPS']+2, cfg['TRAIN']['DINO']['WARMUP_TEACHER_TEMP'], cfg['TRAIN']['DINO']['TEACHER_TEMP'], cfg['TRAIN']['DINO']['WARMUP_TEACHER_EPOCHS'], epochs).to(device)
    optimizer = SGD(student.parameters(), lr=cfg['TRAIN']['LR'])
    scheduler = get_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    writer = SummaryWriter(save_dir / 'logs')

    iters_per_epoch = int(len(dataset)) / cfg['TRAIN']['BATCH_SIZE']

    for epoch in range(1, epochs+1):
        student.train()
        
        if ddp_enable:
            dataloader.sampler.set_epoch(epoch)

        train_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")
        
        for iter, (images, _) in pbar:
            images = [image.to(device) for image in images]

            with autocast(enabled=cfg['TRAIN']['AMP']):
                teacher_pred = teacher(images[:2])  # only 2 global views pass through the teacher
                student_pred = student(images)
                loss = loss_fn(student_pred, teacher_pred, epoch)

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                for p, q in zip(student.module.parameters(), teacher.module.parameters()):
                    q.data.mul_(cfg['TRAIN']['DINO']['TEACHER_MOMENTUM']).add_((1 - cfg['TRAIN']['DINO']['TEACHER_MOMENTUM']) * p.detach().data)

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() * images[0].shape[0]

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= len(dataset)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()

        scheduler.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        torch.save({
            "student": student.module.state_dict() if ddp_enable else student.state_dict(),
            "teacher": teacher.state_dict()
        }, f"{cfg['METHOD']}_{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}_checkpoint.pth")
        torch.save(
            teacher.state_dict(), 
            f"{cfg['METHOD']}_{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.pth"
        )

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