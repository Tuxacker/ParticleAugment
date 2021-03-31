import torch.nn as nn

def loss_from_config(config):
    return nn.CrossEntropyLoss()