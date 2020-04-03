#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import random
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import copy
import os
from PIL import Image

from utilities import getDev, setSeed

avDev = getDev()
print(avDev)

seed = 1
setSeed(seed)

from utilities import showImages, showDetectedImages

lr = 0.001
batchSize = 20
epochs = 100


def transformations(listTransforms):
    return transforms.Compose(listTransforms)


class CudaVisionDataset(Dataset):

    def __init__(self, dir_path, task="detection", transform=None):
        super(CudaVisionDataset, self).__init__()
        self.img_paths, self.target_paths = read_files(img_dir_path=dir_path)
        self.transform = transform
        self.task = task

    def __getitem__(self, index):

        input_img = Image.open(self.img_paths[index])

        # print(input_img.size)
        # showImages(input_img)
        input_img = transforms.functional.resize(input_img, (480, 640))
        # showImages(input_img)
        target_img = Image.open(self.target_paths[index])

        target_img = transforms.functional.resize(target_img, (120, 160))
        if self.transform != None:

            trnfm_input = transformations(self.transform)
            input_img = trnfm_input(input_img)
            target_transform = copy.copy(self.transform)
            if self.task == "segmentation":
                target_transform.pop()
                trnfm_target = transformations(target_transform)
                target_img1 = trnfm_target(target_img)
                target_img_temp = torch.squeeze(target_img1)
                target_img = torch.ones([120, 160], dtype=torch.float64)
                values = torch.tensor([0.0000, 1.0, 2.0, 3.0])
                target_img_temp = target_img_temp * 255
                index = (target_img_temp == values[0]).nonzero()
                target_img[index[:, 0], index[:, 1]] = 0
                index = (target_img_temp == values[1]).nonzero()
                target_img[index[:, 0], index[:, 1]] = 2
                index = (target_img_temp == values[2]).nonzero()
                target_img[index[:, 0], index[:, 1]] = 1
                index = (target_img_temp == values[3]).nonzero()
                target_img[index[:, 0], index[:, 1]] = 2
            else:
                trnfm_target = transformations(target_transform)
                target_img = trnfm_target(target_img)

            # print(target_img.shape)
        return input_img, target_img

    def __len__(self):
        return len(self.target_paths)


def read_files(img_dir_path):
    img_paths = []
    target_paths = []
    if os.path.isdir(img_dir_path):
        print("Folder exists. Reading..")
    dir = os.path.join(img_dir_path, 'input')
    for r, _, f in os.walk(dir):
        f.sort()
        for file in f:
            img_paths.append(os.path.join(r, file))

    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")
    dir = os.path.join(img_dir_path, 'output')
    for r, _, f in os.walk(dir):
        f.sort()
        for file in f:
            target_paths.append(os.path.join(r, file))
    if len(target_paths) == 0:
        print("No Images in given path available. Check directory or format.")

    return img_paths, target_paths


listTransforms_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                    std=[0.5, 0.5, 0.5])]
listTransforms_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])]


class CudaVisionDataLoader:

    def __call__(self, dir_path='./small_data', task="detection", transform=None, batch_size=20):
        dataset = CudaVisionDataset(dir_path, task, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


data = CudaVisionDataLoader()
parentDir = './small_data'
dirDetectionDataset = os.path.join(parentDir, 'detection')
train_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'train'), "detection", listTransforms_train,
                                       batchSize)
validate_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'validate'), "detection",
                                          listTransforms_test, batchSize)
test_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'test'), "detection", listTransforms_test,
                                      batchSize)
dataiter_detection = train_loader_detection.__iter__()

from model import soccerSegment
from metrics import det_accuracy


def train():
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    model = soccerSegment(resnet18, [5, 6, 7, 8], [64, 128, 256, 256, 0], [512, 256, 256, 128], [512, 512, 256], 256)
    model.to(avDev)

    criterionDetected = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = './checkpoints/checkpoint'

    val_patience = 5
    val_counter = 0
    best_val_acc = np.NINF
    epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Checkpoint Loaded")

    while epoch < epochs:

        print("Epoch: ", epoch)
        train_acc = 0.0
        model.train()
        steps = 0
        for images, targets in train_loader_detection:
            images = images.to(avDev)
            targets = targets.to(avDev)
            optimizer.zero_grad()
            segmented, detected = model(images)
            loss = criterionDetected(detected, targets)
            print("Epoch: ", epoch, "step#: ", steps, " Train loss: ", loss.item())
            steps += 1

            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            train_acc += det_accuracy()
            break
        train_acc /= len(train_loader_detection)
        val_acc = 0.0
        model.eval()
        steps = 0
        for images, targets in validate_loader_detection:
            with torch.no_grad():
                images = images.to(avDev)
                targets = targets.to(avDev)
                segmented, detected = model(images)

            segmented, detected = model(images)
            loss = criterionDetected(detected, targets)
            print("Epoch: ", epoch, "step#: ", steps, " Validate loss: ", loss.item())
            steps += 1
            val_acc += det_accuracy()
        val_acc /= len(train_loader_detection)
        if val_acc > best_val_acc:
            val_counter = 0
            best_val_acc = val_acc
            print("New Best Validation Accuracy: ", best_val_acc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
        elif val_counter < val_patience:
            val_counter += 1
            print("Best Validation Accuracy Not Updated ")
        else:
            print("Full Patience Reached ")
            break
        epoch += 1

    count = 0
    for images, targets in test_loader_detection:
        count += 1
        model.eval()
        with torch.no_grad():
            images = images.to(avDev)
            targets = targets.to(avDev)

            segmented, detected = model(images)

            loss = criterionDetected(targets, detected)
            print("Test loss: ", loss.item())
            showImages(images[0], count)
            showDetectedImages(detected[0], count, "output")
            showDetectedImages(targets[0], count, "truth")


train()
