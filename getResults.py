
import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import os
from tvloss import TVLossDetect,TVLossSegment

from utilities import getDev, setSeed ,plot_learning_curve,plot_confusion_matrix
from operator import add

avDev = getDev()

seed = 1
setSeed(seed)

from utilities import showImagesDetection,showImagesSegmentation, showDetectedImages,visualiseSegmented
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
from metrics import segmentationAccuracy,det_metrics,seg_iou,get_predected_centers,get_colored_image,det_confusion_matrix,seg_confusion_matrix
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
    epoch = checkpoint['epoch'] + 1
    seglosses = checkpoint['seglosses']
    detlosses = checkpoint['detlosses']
    valseglosses = checkpoint['valseglosses']
    valdetlosses = checkpoint['valdetlosses']
    print("Checkpoint Loaded")

plot_learning_curve(detlosses,valdetlosses, "detection")
plot_learning_curve(seglosses,valseglosses, "segmentation")
num =0
count =0
color_classes = [[255, 0, 0], [0, 0, 255], [0, 255, 0]]
det_test_metric = np.zeros((5, len(color_classes)))
confusiondet = np.zeros((4,4))
confusionseg = np.zeros((3, 3))
for images, targets, target_center in test_loader_detection:
    count += 1
    model.eval()
    with torch.no_grad():
        images = images.to(avDev)
        targets = targets.to(avDev)

        segmented, detected = model(images)

        tvLoss = TVLossDetect(tvweightdetection)
        total_variation_loss = tvLoss.forward(detected)
        mseloss = criterionDetected(detected, targets)
        loss = mseloss + total_variation_loss
        print("Detection Test loss: ", loss.item())
        ground_truth_centers = get_predected_centers(target_center)
        colored_images = get_colored_image(detected)
        det_test_metric += det_metrics(ground_truth_centers, colored_images, color_classes)
        confusiondet += det_confusion_matrix(ground_truth_centers, colored_images, color_classes)
        for j in range(batchSize):
            num +=1
            showImagesDetection(images[j], num)
            showDetectedImages(detected[j], num, "output")
            showDetectedImages(targets[j], num, "truth")

det_test_metric /= len(test_loader_detection)
plot_confusion_matrix(confusiondet,"detection")
print('Test Detection Overall Accuracy: {}.', np.average(det_test_metric[0]))
print('Ball Accuracy:', det_test_metric[0][0])
print('Goal Pillar Accuracy:', det_test_metric[0][1])
print('Robot Accuracy:', det_test_metric[0][2])

print('Test Detection Overall Recall: {}.', np.average(det_test_metric[1]))
print('Ball Recall:', det_test_metric[1][0])
print('Goal Pillar Recall:', det_test_metric[1][1])
print('Robot Recall:', det_test_metric[1][2])

print('Test Detection Overall Precision: {}.', np.average(det_test_metric[2]))
print('Ball Precision:', det_test_metric[2][0])
print('Goal Pillar Precision:', det_test_metric[2][1])
print('Robot Precision:', det_test_metric[2][2])

print('Test Detection Overall F1score: {}.', np.average(det_test_metric[3]))
print('Ball F1score:', det_test_metric[3][0])
print('Goal Pillar F1score:', det_test_metric[3][1])
print('Robot F1score:', det_test_metric[3][2])

print('Test Detection Overall False Rate: {}.', np.average(det_test_metric[4]))
print('Ball False Rate:', det_test_metric[4][0])
print('Goal Pillar False Rate:', det_test_metric[4][1])
print('Robot False Rate:', det_test_metric[4][2])
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
        confusionseg = seg_confusion_matrix(targets,segmentedLabels)
        accuracies_returned = segmentationAccuracy(segmentedLabels.long(),targets,[0,1,2])
        iou_returned = seg_iou(targets,segmentedLabels.long(),[0,1,2])
        accuracies = list( map(add, accuracies, accuracies_returned) )
        iou = list( map(add, iou, iou_returned) )
        for j in range(batchSize):
            num+=1
            showImagesSegmentation(images[j],num)
            visualiseSegmented(segmentedLabels[j],num,"output")
            visualiseSegmented(targets[j],num,"truth")
    plot_confusion_matrix(confusionseg,"segmentation")   
    print('Test Segmentation Accuracy: {}.',accuracies[3]/count)
    print('Field Accuracy:',accuracies[2]/count)
    print('Line Accuracy:',accuracies[1]/count)
    print('Background Accuracy:',accuracies[0]/count)
    print('Iou:' , iou[3]/count)
    print('Iou Field',iou[2]/count)
    print('Iou Line',iou[1]/count)
    print('Iou Background',iou[0]/count)

    
