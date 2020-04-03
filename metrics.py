
import numpy as np

def seg_iou(ground_truth, predicted, classes):
    avg_iou = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        union = np.logical_or(gt, pr)
        iou_score = np.sum(intersection) / np.sum(union)

        avg_iou += iou_score
    return avg_iou
def det_recall(ground_truth, predicted, classes, tolerance=0.00):
    avg_recall = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        recall_score = np.sum(intersection) / (np.sum(gt) + tolerance)

        avg_recall += recall_score
    return avg_recall


def det_precision(ground_truth, predicted, classes, tolerance=0.00):
    avg_precision = 0
    for c in classes:
        gt = np.all(ground_truth == c, axis=-1)
        pr = np.all(predicted == c, axis=-1)
        intersection = np.logical_and(gt, pr)
        precision_score = np.sum(intersection) / (np.sum(pr) + tolerance)

        avg_precision += precision_score
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
    return avg_false_rate
