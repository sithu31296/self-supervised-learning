from .schedulers import *
from .loss import *

schs = {
    "steplr": StepLR
}

losses = {
    "dinoloss": DINOLoss,
    "ddinoloss": DDINOLoss
}

def get_loss(cfg, epochs):
    loss_fn_name = cfg['TRAIN']['LOSS']
    assert loss_fn_name in losses.keys()
    return losses[loss_fn_name](cfg['TRAIN']['DINO']['HEAD_DIM'], cfg['TRAIN']['DINO']['LOCAL_CROPS']+2, cfg['TRAIN']['DINO']['WARMUP_TEACHER_TEMP'], cfg['TRAIN']['DINO']['TEACHER_TEMP'], cfg['TRAIN']['DINO']['WARMUP_TEACHER_EPOCHS'], epochs)


def get_scheduler(cfg, optimizer):
    scheduler_name = cfg['TRAIN']['SCHEDULER']['NAME']
    assert scheduler_name in schs.keys(), f"Unavailable scheduler name >> {scheduler_name}.\nList of available schedulers: {list(schs.keys())}"
    return schs[scheduler_name](optimizer, *cfg['TRAIN']['SCHEDULER']['PARAMS'])