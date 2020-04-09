# !/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import os

from utilities import getDev, setSeed

avDev = getDev()
print(avDev)
avDev = "cpu"
seed = 1
setSeed(seed)

lr = 0.001
batchSize = 20
epochs = 100
from dataloader import CudaVisionDataLoader

listTransforms_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                    std=[0.5, 0.5, 0.5])]
listTransforms_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])]

data = CudaVisionDataLoader()
parentDir = './example_data'
dirDetectionDataset = os.path.join(parentDir, 'detection')
# train_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'train'), "detection", listTransforms_train,
#                                        batchSize)
validate_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'validate'), "detection",
                                          listTransforms_test, batchSize)
# test_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'test'), "detection", listTransforms_test,
#                                       batchSize)

dirSegmentationDataset = os.path.join(parentDir, 'segmentation')

# train_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'train'), "segmentation",
#                                           listTransforms_train, batchSize)
validate_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'validate'), "segmentation",
                                             listTransforms_test, batchSize)
# test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'test'), "segmentation",
#                                          listTransforms_test, batchSize)

from model import soccerSegment


def train():
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    model = soccerSegment(resnet18, [5, 6, 7, 8], [64, 128, 256, 256, 0], [512, 256, 256, 128], [512, 512, 256], 256)
    model.to(avDev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = './checkpoints/checkpoint'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint Loaded")
    count = 0
    from utilities import showImagesDetection, visualiseSegmented, showDetectedImages
    image_range = [0, 1]

    model.eval()
    for images, targets in validate_loader_segmentation:

        if not (count >= image_range[0] or count < image_range[1]):
            continue

        with torch.no_grad():
            images = images.to(avDev)
            segmented, detected = model(images)

            num = 0

            for j in range(batchSize):
                num += 1
                showImagesDetection(images[j], num)
                showDetectedImages(detected[j], num, "output_detection")
                segmentedLabels = torch.argmax(segmented, dim=1)
                visualiseSegmented(segmentedLabels[j], num, "output_segmentation")
        count += 1


train()
