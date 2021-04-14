import torch

def get_default_device():
    """ Using GPU if available or CPU """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

default_device = get_default_device()