from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
import os
import torch
from torch.utils.data.dataloader import DataLoader
from utils import constants, get_default_device
from image_folder_with_path import ImageFolderWithPaths

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



class DeviceDataLoader():
    """ wrap a Dataloader to move data to a device """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """ yield a batch of data after moving it to device """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """ return number of batch size """
        return len(self.dl)


default_device = get_default_device.default_device

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.uniform(5, 10)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

classes = os.listdir(constants.DATA_PATH + constants.TRAIN_PATH)

training_dataset = ImageFolder(constants.DATA_PATH + constants.TRAIN_PATH, transform=train_transforms)
valid_dataset = ImageFolder(constants.DATA_PATH + constants.VAL_PATH, transform=test_transforms)
# testing_dataset = ImageFolder(constants.DATA_PATH + constants.TEST_PATH, transform=test_transforms)

# training_dataset = ImageFolderWithPaths(constants.DATA_PATH + constants.TRAIN_PATH, transform=train_transforms)
# valid_dataset = ImageFolderWithPaths(constants.DATA_PATH + constants.VAL_PATH, transform=test_transforms)
testing_dataset = ImageFolderWithPaths(constants.DATA_PATH + constants.TEST_PATH, transform=test_transforms)


torch.manual_seed(constants.RANDOM_SEED)

train_dl = DataLoader(training_dataset, constants.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_dl = DataLoader(valid_dataset, constants.BATCH_SIZE, num_workers=8, pin_memory=True)
test_dl = DataLoader(testing_dataset, constants.BATCH_SIZE, num_workers=8, pin_memory=True)


"""
Now we can wrap our training and validation data loaders using DeviceDataLoader for automatically transferring batches
of data to GPU (if available), and use to_device to move our model to GPU (if available)
"""
train_dl = DeviceDataLoader(train_dl, default_device)
val_dl = DeviceDataLoader(val_dl, default_device)
test_dl = DeviceDataLoader(test_dl, default_device)