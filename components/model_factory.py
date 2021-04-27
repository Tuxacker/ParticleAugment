from .wide_resnet import Wide_ResNet
from .shake_resnet import ShakeResNet

def model_from_config(config):

    if not hasattr(config, "model"):
        raise ValueError("NN model not set!")
    
    if config.model == "wide_resnet":

        depth = config.model_params.depth
        width = config.model_params.width
        dropout_rate = config.model_params.dropout_rate
        num_classes = config.dataset_params.num_classes

        return Wide_ResNet(depth, width, dropout_rate=dropout_rate, num_classes=num_classes)

    elif config.model == "shake_shake":

        depth = config.model_params.depth
        width = config.model_params.width
        num_classes = config.dataset_params.num_classes

        return ShakeResNet(depth, width, num_classes)

    else:
        raise ValueError("{} is not a valid model identifier!".format(config.model))