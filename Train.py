import torch
import torchvision.models as models
from utils import constants, get_default_device
from data_loader import to_device, train_dl, val_dl, testing_dataset, training_dataset
from utils.visualization import plot_losses, plot_accuracies, plot_lrs
from fit import fit_one_cycle
from utils.helper import evaluate
from model.MobileNet import MobileNetV2
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
default_device = get_default_device.default_device

def training():
    model_MobileNetV1= to_device(MobileNetV2(3, constants.NUM_OF_FEATURES), default_device)
    opt_func = torch.optim.Adam
    history6 = fit_one_cycle(constants.NUM_OF_EPOCHS, constants.MAX_LEARNING_RATE, model_MobileNetV1, train_dl, val_dl,
                             grad_clip=constants.GRAD_CLIP,
                             weight_decay=constants.WEIGHT_DECAY,
                             opt_func=opt_func)
    plot_accuracies(history6)
    plot_losses(history6)
    plot_lrs(history6)

    result = evaluate(model_MobileNetV1, val_dl)
    print(result)

training()