import math

import torch.optim.lr_scheduler as lrs

def lrs_from_config(config, optimizer):
    if config.lrs_params.warmup:
        lambda_warmup = lambda e: (e + 1) / float(config.lrs_params.warmup_epochs) if e < config.lrs_params.warmup_epochs else 0.5 + 0.5 * math.cos(math.pi * e / config.epochs)
        return lrs.LambdaLR(optimizer, lambda_warmup)
    else:
        return lrs.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.0)