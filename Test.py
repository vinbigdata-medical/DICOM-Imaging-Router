import torch
import matplotlib.pyplot as plt
from utils import constants, get_default_device
from sklearn.metrics import confusion_matrix
from data_loader import to_device, train_dl, val_dl, testing_dataset, training_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import time 
# from sklearn.metrics import roc_curve
from model.MobileNet import MobileNetV1
from model.MobileNet import MobileNetV2
from model.ResNet import ResNet34
from model.EfficientNet import EfficientNet
from torchvision import models
from PIL import ImageFile
import os
import shutil
import numpy as np 
import shutil
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
default_device = torch.device("cuda")

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), default_device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return training_dataset.classes[preds[0].item()]

# luu y khi chay de y constants
def predict():
    model = MobileNetV1(image_channels=constants.image_channels,num_classes=constants.NUM_OF_FEATURES)
    model.cuda()
    model.eval()
    checkpoint = torch.load(constants.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_true = []
    y_pred = []
    index = 0

    path_folder = "/home/single1/BACKUP/VanDung/BodyParts/Predicted_Pediatric"
    correct = 0
    wrong_total = 0
    total = len(testing_dataset)
    start_time = time.time()
    for currentImage in testing_dataset:
        index += 1
        image_name = currentImage[-1].split('/')[-1]
        # print(image_name)
        img, label, path = currentImage
        true_label = training_dataset.classes[label]
        pred_label = predict_image(img, model)
        y_true.append(true_label)
        y_pred.append(pred_label)
        if pred_label == 'pediatric':
            shutil.copyfile(path, f'{path_folder}/{image_name}')

        if pred_label == true_label:
            correct += 1
        else:
            wrong_total += 1
        #     newpath = "./predicted" + path.split('dataset')[-1]
        #     os.rename(path, newpath)
        #     plt.imshow(img.permute(1, 2, 0))
        #     plt.text(5, 5, 'ground truth: ' + true_label + ', predicted: ' + pred_label , bbox={'facecolor': 'white', 'pad': 5})
        #     plt.show()

        item = ('index: ', index, 'Label:', true_label, ', Predicted:', pred_label)
        print(item)
        stop_time = time.time()
    matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    F1_score = f1_score(y_true, y_pred, average='macro')
    print('precision = ', precision)
    print('recall = ', recall)
    print('F1_score = ', F1_score)
predict()