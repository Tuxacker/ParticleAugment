import torch.optim as optim

def optimizer_from_config(config, model):

    lr = config.lr
    momentum = config.optimizer_params.momentum
    weight_decay = config.optimizer_params.weight_decay
    nesterov = config.optimizer_params.nesterov

    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)