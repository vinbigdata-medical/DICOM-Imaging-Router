import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
