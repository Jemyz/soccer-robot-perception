#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import os
from tvloss import TVLossDetect

from utilities import getDev, setSeed

avDev = getDev()
print(avDev)
avDev = "cpu"
seed = 1
setSeed(seed)

from utilities import showImages, showDetectedImages

lr = 0.001
batchSize = 20
epochs = 100


from dataloader import CudaVisionDataLoader


listTransforms_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                    std=[0.5, 0.5, 0.5])]
listTransforms_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])]



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
            showDetectedImages(targets[0],0,"train")
            optimizer.zero_grad()
            segmented, detected = model(images)
            tvLoss = TVLossDetect()
            total_variation_loss = tvLoss.forward(detected)
            mse_loss = criterionDetected(detected, targets)
            loss =  mse_loss + total_variation_loss
            print("Epoch: ", epoch, "step#: ", steps, " Train loss: ", loss.item())
            steps += 1

            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            train_acc += det_accuracy()
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
            tvLoss = TVLossDetect()
            total_variation_loss = tvLoss.forward(detected)
            mse_loss = criterionDetected(detected, targets)
            loss =  mse_loss + total_variation_loss
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

            tvLoss = TVLossDetect()
            total_variation_loss = tvLoss.forward(detected)
            mse_loss = criterionDetected(detected, targets)
            loss =  mse_loss + total_variation_loss
            print("Test loss: ", loss.item())
            showImages(images[0], count)
            showDetectedImages(detected[0], count, "output")
            showDetectedImages(targets[0], count, "truth")


train()
