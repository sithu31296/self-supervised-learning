from .schedulers import *

schs = {
    "steplr": StepLR
}


def get_scheduler(cfg, optimizer):
    scheduler_name = cfg['TRAIN']['SCHEDULER']['NAME']
    assert scheduler_name in schs.keys(), f"Unavailable scheduler name >> {scheduler_name}.\nList of available schedulers: {list(schs.keys())}"
    return schs[scheduler_name](optimizer, *cfg['TRAIN']['SCHEDULER']['PARAMS'])