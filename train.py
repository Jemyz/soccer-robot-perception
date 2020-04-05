#!/usr/bin/env python
# coding: utf-8

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
batchSize = 20
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

train_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'train'), "detection", listTransforms_train,
                                       batchSize)
validate_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'validate'), "detection",
                                          listTransforms_test, batchSize)
test_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'test'), "detection", listTransforms_test,
                                      batchSize)
train_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'train'),"segmentation",listTransforms_train,batchSize)
validate_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'validate'),"segmentation",listTransforms_test,batchSize)
test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'test'),"segmentation",listTransforms_test,batchSize)


from model import soccerSegment
from metrics import det_accuracy,segmentationAccuracy,seg_iou


def train():
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
    epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        segloss = checkpoint['seg_loss']
        detloss = checkpoint['det_loss']
        print("Checkpoint Loaded")
    detlosses= []
    seglosses = []
    while epoch < epochs:

        print("Epoch: ", epoch)
        train_acc = 0.0
        model.train()
        steps = 0
        fullDataCovered = 0
        dataiter_detection = train_loader_detection.__iter__()
        dataiter_segmentation = train_loader_segmentation.__iter__()
        
        while(fullDataCovered!=1):
            for i in range(2):
                try:
                    images, targets,list_center,list_class = dataiter_detection.next()
                except:
                    fullDataCovered = 1
                    dataiter_detection = train_loader_detection.__iter__()
                    images, targets,list_center,list_class = dataiter_detection.next()
                images = images.to(avDev)
                targets = targets.to(avDev)
                optimizer.zero_grad()
                segmented, detected = model(images)
                tvLoss = TVLossDetect(tvweightdetection)
                total_variation_loss = tvLoss.forward(detected)
                mseloss = criterionDetected(detected, targets)
                loss =  mseloss + total_variation_loss
                detlosses.append(loss.item())
                print("Epoch: ", epoch, "step#: ", steps, " Train detection loss: ", loss.item())
                steps += 1
                    # Getting gradients w.r.t. parameters
                loss.backward()
            # Updating parameters
                optimizer.step()
                
                #train_acc += det_accuracy(detImage,colorMap)
                i +=1
            try:
                    images, targets = dataiter_segmentation.next()
            except:
                    fullDataCovered = 1
                    dataiter_segmentation = train_loader_segmentation.__iter__()
                    images, targets = dataiter_segmentation.next()
            images = images.to(avDev)
            targets = targets.to(avDev)
            optimizer.zero_grad()
            segmented, detected = model(images)
            segmentedLabels=torch.argmax(segmented,dim=1)
            accuracies =segmentationAccuracy(segmentedLabels.long(),targets,[0,1,2])
            tvLoss = TVLossSegment(tvweightsegmentation)
            total_variation_loss = tvLoss.forward(segmented)
            cross_entropy_loss = criterionSegmented(segmented, targets)
            loss =  cross_entropy_loss + total_variation_loss
            seglosses.append(loss.item())
            print("Epoch: ", epoch, "step#: ", steps, " Train segmentation loss: ", loss.item())
            steps += 1
                    # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            train_acc += det_accuracy()
            i +=1
            
        train_acc /= len(train_loader_detection)
        val_acc = 0.0
        seg_acc = 0.0
        model.eval()
        steps = 0
        
        for images, targets,list_center,list_class in validate_loader_detection:
            with torch.no_grad():
                images = images.to(avDev)
                targets = targets.to(avDev)
                segmented, detected = model(images)

                segmented, detected = model(images)
                tvLoss = TVLossDetect(tvweightdetection)
                total_variation_loss = tvLoss.forward(detected)
                mseloss = criterionDetected(detected, targets)
                detloss =  mseloss + total_variation_loss
                
                print("Epoch: ", epoch, "step#: ", steps, " Detection Validate loss: ", detloss.item())
                steps += 1
                
        val_acc /= len(validate_loader_detection)
        for images, targets in validate_loader_segmentation:
            model.eval()
            with torch.no_grad():
                         
                images = images.to(avDev)
                targets = targets.to(avDev)

                segmented,detected = model(images)

                segmentedLabels=torch.argmax(segmented,dim=1)
                tvLoss = TVLossSegment(tvweightsegmentation)
            total_variation_loss = tvLoss.forward(segmented)
            entropy_loss = criterionSegmented(segmented, targets.long())
            segloss = entropy_loss + total_variation_loss
            print("Epoch: ", epoch, "step#: ", steps, " Segmentation Validate loss: ", loss.item())
            
            accuracies =segmentationAccuracy(segmentedLabels.long(),targets,[0,1,2])
            print('Validate Segmentation Accuracy: Background,Line,Field,Total:',accuracies[0],accuracies[1],accuracies[2],accuracies[3])
            seg_acc += accuracies[3]
        seg_acc /=len(validate_loader_segmentation) 
        acc = (val_acc+seg_acc)/2
        if acc > best_acc:
            val_counter = 0
            best_acc = acc
            print("New Best Validation Accuracy Detection,Segmentation: ", val_acc,seg_acc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'det_loss': detloss,
                'seg_loss': segloss
            }, checkpoint_path)
        elif val_counter < val_patience:
            val_counter += 1
            print("Best Validation Accuracy Not Updated ")
        else:
            print("Full Patience Reached ")
            break
        epoch += 1

    count = 0
    plot_learning_curve(detlosses,"detection")
    plot_learning_curve(seglosses,"segmentation")
    for images, targets,list_center,list_class in test_loader_detection:
        count += 1
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
                showImages(images[j], count+j)
                showDetectedImages(detected[j], count+j, "output")
                showDetectedImages(targets[j], count+j, "truth")
    accuracies = [0,0,0,0]
    iou = [0,0,0,0]
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
            showImages(images[j],count+j)
            visualiseSegmented(segmentedLabels[j],count+j,"output")
            visualiseSegmented(targets[j],count+j,"truth")
        
    print('Test Segmentation Accuracy: {}.',accuracies[3]/count)
    print('Field Accuracy:',accuracies[2]/count)
    print('Line Accuracy:',accuracies[1]/count)
    print('Background Accuracy:',accuracies[0]/count)
    print('Iou:' , iou[3]/count)
    print('Iou Field',iou[2]/count)
    print('Iou Line',iou[1]/count)
    print('Iou Background',iou[0]/count)
    

train()
