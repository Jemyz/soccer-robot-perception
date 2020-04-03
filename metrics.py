import numpy as np


def segmentationAccuracy(segmented, targets, classes):
    accuracies = [0, 0, 0, 0]
    avg_acc = 0
    for c in classes:
        total_pixels = (targets == c).sum()
        correct_pixels = np.logical_and((targets == c), (segmented == c)).sum()
        accuracies[c] = correct_pixels / total_pixels.item() * 100
        avg_acc = avg_acc + accuracies[c]
    avg_acc = avg_acc / len(classes)
    accuracies[3] = avg_acc
    return accuracies


def seg_iou(ground_truth, predicted, classes):
    avg_iou = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        union = np.logical_or(gt, pr)
        iou_score = np.sum(intersection) / np.sum(union)

        avg_iou += iou_score
    avg_iou = avg_iou / len(classes)
    return avg_iou


def det_accuracy():
    return 0.0


def det_recall(ground_truth, predicted, classes, tolerance=0.00):
    avg_recall = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        recall_score = np.sum(intersection) / (np.sum(gt) + tolerance)

        avg_recall += recall_score
    avg_recall = avg_recall / len(classes)
    return avg_recall


def det_precision(ground_truth, predicted, classes, tolerance=0.00):
    avg_precision = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        precision_score = np.sum(intersection) / (np.sum(pr) + tolerance)

        avg_precision += precision_score
    avg_precision = avg_precision / len(classes)
    return avg_precision


def det_f1score(precision, recall, tolerance=0.00):
    return (precision * recall) / (precision + recall + tolerance)


def det_false_rate(ground_truth, predicted, classes, tolerance=0.00):
    avg_false_rate = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        union = np.logical_or(gt, pr)
        false_rate_score = 1 - (np.sum(intersection) / (np.sum(union) + tolerance))

        avg_false_rate += false_rate_score
    avg_false_rate = avg_false_rate / len(classes)
    return avg_false_rate
