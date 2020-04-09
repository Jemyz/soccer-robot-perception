import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import os
from tvloss import TVLossDetect, TVLossSegment

from utilities import getDev, setSeed, plot_learning_curve
from operator import add



def train(lr,batchSize,epochs,tvweightdetection,tvweightsegmentation,saveImages):
    avDev = getDev()
    print(avDev)

    seed = 1
    setSeed(seed)

    from utilities import showImagesDetection,showImagesSegmentation, showDetectedImages, visualiseSegmented


    from dataloader import CudaVisionDataLoader

    listTransforms_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                        std=[0.5, 0.5, 0.5])]
    listTransforms_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                       std=[0.5, 0.5, 0.5])]

    data = CudaVisionDataLoader()
    parentDir = './small_data'
    dirDetectionDataset = os.path.join(parentDir, 'detection')
    dirSegmentationDataset = os.path.join(parentDir, 'segmentation')

    train_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'train'), "detection", listTransforms_train,
                                           batchSize)
    validate_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'validate'), "detection",
                                              listTransforms_test, batchSize)
    test_loader_detection = data.__call__(os.path.join(dirDetectionDataset, 'test'), "detection", listTransforms_test,
                                          batchSize)
    train_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'train'), "segmentation",
                                              listTransforms_train, batchSize)
    validate_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'validate'), "segmentation",
                                                 listTransforms_test, batchSize)
    test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset, 'test'), "segmentation",
                                             listTransforms_test, batchSize)

    from model import soccerSegment
    from metrics import det_metrics, segmentationAccuracy, seg_iou,det_confusion_matrix,seg_confusion_matrix
    from utilities import get_colored_image, get_predected_centers,plot_confusion_matrix

    import torchvision.models as models
    print("Loading pretrained ResNet18")
    resnet18 = models.resnet18(pretrained=True)
    model = soccerSegment(resnet18, [5, 6, 7, 8], [64, 128, 256, 256, 0], [512, 256, 256, 128], [512, 512, 256], 256)
    model.to(avDev)

    criterionDetected = nn.MSELoss()
    criterionSegmented = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = './checkpoints/checkpoint'

    val_patience = 10
    val_counter = 0
    best_acc = np.NINF
    epoch = 0
    color_classes = [[255, 0, 0], [0, 0, 255], [0, 255, 0]]
    detlosses = []
    seglosses = []
    valdetlosses = []
    valseglosses = []
    print("Load saved model if any")
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
    
    print("Specified epochs for training: ",epoch," The patience value for early stopping: ",val_patience,
          "Batch Size: ",batchSize," Learning Rate: ",lr)
    print("Starting training")
    while epoch < epochs:
        detlosstrain = 0
        seglosstrain = 0
        print("Epoch: ", epoch)
        det_train_acc = 0.0
        seg_train_acc = 0.0

        model.train()
        steps = 0
        fullDataCovered = 0
        dataiter_detection = train_loader_detection.__iter__()
        dataiter_segmentation = train_loader_segmentation.__iter__()
        
        while (fullDataCovered != 1):
            for i in range(4):
                try:
                    images, targets, target_centers = dataiter_detection.next()
                except:
                    fullDataCovered = 1
                    dataiter_detection = train_loader_detection.__iter__()
                    images, targets, target_centers = dataiter_detection.next()
                images = images.to(avDev)
                targets = targets.to(avDev)
                optimizer.zero_grad()
                segmented, detected = model(images)
                tvLoss = TVLossDetect(tvweightdetection)
                total_variation_loss = tvLoss.forward(detected)
                mseloss = criterionDetected(detected, targets)
                loss = mseloss + total_variation_loss
                detlosstrain += loss.item()
                steps += 1
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()

                ground_truth_centers = get_predected_centers(target_centers)
                colored_images = get_colored_image(detected)
                det_train_acc += np.average(det_metrics(ground_truth_centers, colored_images, color_classes)[0])
                i += 1
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
            segmentedLabels = torch.argmax(segmented, dim=1)
            accuracies = segmentationAccuracy(segmentedLabels.long(), targets, [0, 1, 2])
            tvLoss = TVLossSegment(tvweightsegmentation)
            total_variation_loss = tvLoss.forward(segmented)
            cross_entropy_loss = criterionSegmented(segmented, targets)
            loss = cross_entropy_loss + total_variation_loss
            seglosstrain += loss.item()
            steps += 1
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            seg_train_acc += accuracies[3]
            
        det_train_acc /= len(train_loader_detection)
        seg_train_acc /= len(train_loader_segmentation)
        detlosstrain /= len(train_loader_detection)
        seglosstrain /= len(train_loader_segmentation)
        print("Training finished , starting with validation")
        det_val_acc = 0.0
        seg_val_acc = 0.0
        model.eval()
        steps = 0
        detlosses.append(detlosstrain)
        seglosses.append(seglosstrain)
        detlossval = 0
        seglossval = 0
        for images, targets, target_centers in validate_loader_detection:
            with torch.no_grad():
                images = images.to(avDev)
                targets = targets.to(avDev)

                segmented, detected = model(images)
                tvLoss = TVLossDetect(tvweightdetection)
                total_variation_loss = tvLoss.forward(detected)
                mseloss = criterionDetected(detected, targets)
                detloss = mseloss + total_variation_loss
                detlossval+=detloss.item()
                steps += 1

                ground_truth_centers = get_predected_centers(target_centers)
                colored_images = get_colored_image(detected)
                det_val_acc += np.average(det_metrics(ground_truth_centers, colored_images, color_classes)[0])

        for images, targets in validate_loader_segmentation:
            model.eval()
            with torch.no_grad():
                images = images.to(avDev)
                targets = targets.to(avDev)

                segmented, detected = model(images)

                segmentedLabels = torch.argmax(segmented, dim=1)
                tvLoss = TVLossSegment(tvweightsegmentation)
            total_variation_loss = tvLoss.forward(segmented)
            entropy_loss = criterionSegmented(segmented, targets.long())
            segloss = entropy_loss + total_variation_loss
            seglossval += segloss.item()
            accuracies = segmentationAccuracy(segmentedLabels.long(), targets, [0, 1, 2])
            seg_val_acc += accuracies[3]

        det_val_acc /= len(validate_loader_detection)
        seg_val_acc /= len(validate_loader_segmentation)
        detlossval /= len(validate_loader_detection)
        seglossval /= len(validate_loader_segmentation)
        valdetlosses.append(detlossval)
        valseglosses.append(seglossval)
        print("Epoch: ", epoch, " Finished")
        print("Epoch: ", epoch, " Detection Train Accuracy: ", det_train_acc, " Detection Validation Accuracy: ",
              det_val_acc, "Detection Loss Train:",detlosstrain)
        print("Epoch: ", epoch, " Segmentation Train Accuracy: ", det_train_acc,
              " Segmentation Validation Accuracy: ", seg_val_acc,"Segmentation Train Accuracy:",seglosstrain)

        acc = (det_val_acc + seg_val_acc) / 2
        if acc > best_acc:
            val_counter = 0
            best_acc = acc
            print("New Best Validation Accuracy Detection: ", det_val_acc, " Segmentation: ", seg_val_acc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'detlosses': detlosses,
                'seglosses': seglosses,
                'valdetlosses': valdetlosses,
                'valseglosses':valseglosses
            }, checkpoint_path)
        elif val_counter < val_patience:
            val_counter += 1
            print("Best Validation Accuracy Not Updated ")
        else:
            print("Full Patience Reached ")
            break
        epoch += 1

    count = 0
    num = 0
    plot_learning_curve(detlosses,valdetlosses, "detection")
    plot_learning_curve(seglosses,valseglosses, "segmentation")
    det_test_metric = np.zeros((5, len(color_classes)))
    confusiondet = np.zeros((4,4))
    confusionseg = np.zeros((3, 3))
    print("Training finished starting with testdatatset")
    print("Starting with detection")
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
            if(saveImages):
                for j in range(batchSize):
                    num +=1
                    showImagesDetection(images[j], num)
                    showDetectedImages(detected[j], num, "output")
                    showDetectedImages(targets[j], num, "truth")

    det_test_metric /= len(test_loader_detection)
    plot_confusion_matrix(confusiondet,"detection")
    print('Test Detection Overall Accuracy: {}.', np.average(det_test_metric[0]))
    print('Ball Accuracy:', det_test_metric[0][0])
    print('Goal Accuracy:', det_test_metric[0][1])
    print('Robot Pillar Accuracy:', det_test_metric[0][2])

    print('Test Detection Overall Recall: {}.', np.average(det_test_metric[1]))
    print('Ball Recall:', det_test_metric[1][0])
    print('Goal Recall:', det_test_metric[1][1])
    print('Robot Pillar Recall:', det_test_metric[1][2])

    print('Test Detection Overall Precision: {}.', np.average(det_test_metric[2]))
    print('Ball Precision:', det_test_metric[2][0])
    print('Goal Precision:', det_test_metric[2][1])
    print('Robot Pillar Precision:', det_test_metric[2][2])

    print('Test Detection Overall F1score: {}.', np.average(det_test_metric[3]))
    print('Ball F1score:', det_test_metric[3][0])
    print('Goal F1score:', det_test_metric[3][1])
    print('Robot Pillar F1score:', det_test_metric[3][2])

    print('Test Detection Overall False Rate: {}.', np.average(det_test_metric[4]))
    print('Ball False Rate:', det_test_metric[4][0])
    print('Goal False Rate:', det_test_metric[4][1])
    print('Robot Pillar False Rate:', det_test_metric[4][2])

    accuracies = [0, 0, 0, 0]
    iou = [0, 0, 0, 0]
    num =0
    count =0
    for images, targets in test_loader_segmentation:
        count += 1
        model.eval()
        with torch.no_grad():

            images = images.to(avDev)
            targets = targets.to(avDev)

            segmented, detected = model(images)

            segmentedLabels = torch.argmax(segmented, dim=1)
            tvLoss = TVLossSegment(tvweightsegmentation)
            total_variation_loss = tvLoss.forward(segmented)
            entropy_loss = criterionSegmented(segmented, targets.long())
            loss = entropy_loss + total_variation_loss
            print("Segmentation Test loss: ", loss.item())
        confusionseg = seg_confusion_matrix(targets,segmentedLabels)
        accuracies_returned = segmentationAccuracy(segmentedLabels.long(), targets, [0, 1, 2])
        iou_returned = seg_iou(targets, segmentedLabels.long(), [0, 1, 2])
        accuracies = list(map(add, accuracies, accuracies_returned))
        iou = list(map(add, iou, iou_returned))
        if(saveImages):
            for j in range(batchSize):
                num+=1
                showImagesSegmentation(images[j],num)
                visualiseSegmented(segmentedLabels[j], num, "output")
                visualiseSegmented(targets[j], num, "truth")
    plot_confusion_matrix(confusionseg,"segmentation")
    print('Test Segmentation Accuracy: {}.', accuracies[3] / count)
    print('Field Accuracy: ', accuracies[2] / count)
    print('Line Accuracy: ', accuracies[1] / count)
    print('Background Accuracy: ', accuracies[0] / count)
    print('Iou: ', iou[3] / count)
    print('Iou Field: ', iou[2] / count)
    print('Iou Line: ', iou[1] / count)
    print('Iou Background: ', iou[0] / count)



