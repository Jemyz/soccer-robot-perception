
import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import os
from tvloss import TVLossDetect,TVLossSegment

from utilities import getDev, setSeed ,plot_learning_curve
from operator import add
avDev = getDev()
print(avDev)
avDev = "cpu"
seed = 1
setSeed(seed)

from utilities import showImages, showDetectedImages,visualiseSegmented
lr = 0.001
batchSize = 1
epochs = 100
tvweightdetection = 0.000001
tvweightsegmentation = 0.00001


from dataloader import CudaVisionDataLoader


listTransforms_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                    std=[0.5, 0.5, 0.5])]
listTransforms_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])]

data = CudaVisionDataLoader()
parentDir = './small_data'
dirDetectionDataset = os.path.join(parentDir, 'detection')
dirSegmentationDataset = os.path.join(parentDir , 'segmentation')

test_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'test'), "detection", listTransforms_test,
                                      batchSize)
test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'test'),"segmentation",listTransforms_test,batchSize)


from model import soccerSegment
from metrics import det_accuracy,segmentationAccuracy,seg_iou
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
model = soccerSegment(resnet18, [5, 6, 7, 8], [64, 128, 256, 256, 0], [512, 256, 256, 128], [512, 512, 256], 256)
model.to(avDev)

criterionDetected = nn.MSELoss()
criterionSegmented = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

checkpoint_path = './checkpoints/checkpoint'

val_patience = 5
val_counter = 0
best_acc = np.NINF
    
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint Loaded")
num =0
for images, targets,list_center,list_class in test_loader_detection:
        
    model.eval()
    with torch.no_grad():
        images = images.to(avDev)
        targets = targets.to(avDev)

        segmented, detected = model(images)

        tvLoss = TVLossDetect(tvweightdetection)
        total_variation_loss = tvLoss.forward(detected)
        mseloss = criterionDetected(detected, targets)
        loss =  mseloss + total_variation_loss
        print("Detection Test loss: ", loss.item())
        for j in range(batchSize):
            num +=1
            showImages(images[j], num)
            showDetectedImages(detected[j], num, "output")
            showDetectedImages(targets[j], num, "truth")
accuracies = [0,0,0,0]
iou = [0,0,0,0]
count = 0
num=0
for images, targets in test_loader_segmentation:
    count+=1
    model.eval()
    with torch.no_grad():
                         
        images = images.to(avDev)
        targets = targets.to(avDev)

        segmented,detected = model(images)

        segmentedLabels=torch.argmax(segmented,dim=1)
        tvLoss = TVLossSegment(tvweightsegmentation)
        total_variation_loss = tvLoss.forward(segmented)
        entropy_loss = criterionSegmented(segmented, targets.long())
        loss = entropy_loss + total_variation_loss
        print("Segmentation Test loss: ", loss.item())
        accuracies_returned = segmentationAccuracy(segmentedLabels.long(),targets,[0,1,2])
        iou_returned = seg_iou(targets,segmentedLabels.long(),[0,1,2])
        accuracies = list( map(add, accuracies, accuracies_returned) )
        iou = list( map(add, iou, iou_returned) )
        for j in range(batchSize):
            num+=1
            showImages(images[j],num)
            visualiseSegmented(segmentedLabels[j],num,"output")
            visualiseSegmented(targets[j],num,"truth")
        
    print('Test Segmentation Accuracy: {}.',accuracies[3]/count)
    print('Field Accuracy:',accuracies[2]/count)
    print('Line Accuracy:',accuracies[1]/count)
    print('Background Accuracy:',accuracies[0]/count)
    print('Iou:' , iou[3]/count)
    print('Iou Field',iou[2]/count)
    print('Iou Line',iou[1]/count)
    print('Iou Background',iou[0]/count)

    
