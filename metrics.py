import numpy as np


def segmentationAccuracy(segmented, targets, classes):
    accuracies = [0, 0, 0, 0]
    avg_acc = 0
    targets = targets.cpu()
    segmented = segmented.cpu()
    for c in classes:
        total_pixels = (targets == c).sum()
        correct_pixels = np.logical_and((targets == c), (segmented == c)).sum()
        accuracies[c] = correct_pixels.item() / total_pixels.item() * 100
        avg_acc = avg_acc + accuracies[c]
    avg_acc = avg_acc / len(classes)
    accuracies[3] = avg_acc
    return accuracies


def seg_iou(ground_truth, predicted, classes):
    avg_iou = 0
    iou = [0,0,0,0]
    for c in classes:
        gt = (ground_truth.cpu()==c)
        pr = (predicted.cpu()==c)
        intersection = np.logical_and(gt, pr)
        union = np.logical_or(gt, pr)
        iou_score = intersection.sum().item() / union.sum().item()
        iou[c] = iou_score
        avg_iou += iou_score
    iou[3]=avg_iou/len(classes)
    return iou

def det_info(ground_truth_centers, predicted_centers, classes, threshold=5):
    tp = 0
    fp = 0
    fn = 0
    for c in classes:
        gts = ground_truth_centers[np.all(np.array(ground_truth_centers[:, 2].tolist()) == c, axis=-1)]
        prs = predicted_centers[np.all(np.array(predicted_centers[:, 2].tolist()) == c, axis=-1)]

        from scipy.spatial import KDTree

        tree = KDTree(gts[:, 0].tolist())
        neighbor_dists, neighbor_indices = tree.query(prs[:, 0].tolist())

        sorted_indices = (np.lexsort((neighbor_dists, neighbor_indices)))
        correct_indices = [sorted_indices[i] for i in
                           range(len(sorted_indices)) if
                           (neighbor_indices[sorted_indices[i]] != neighbor_indices[sorted_indices[i - 1]]
                            and neighbor_dists[sorted_indices[i]] < threshold) or i == 0]

        tp += len(correct_indices)
        fp += len(sorted_indices) - len(correct_indices)
        fn += len(gts) - len(correct_indices)
    return tp, fp, fn


def det_accuracy(ground_truth_centers, predicted_centers, classes, threshold=5):
    tp, fp, fn = det_info(ground_truth_centers, predicted_centers, classes, threshold)
    return tp / (tp + fp + fn)


def det_recall(ground_truth_centers, predicted_centers, classes, threshold=5):
    tp, fp, fn = det_info(ground_truth_centers, predicted_centers, classes, threshold)
    return tp / (tp + fn)


def det_precision(ground_truth_centers, predicted_centers, classes, threshold=5):
    tp, fp, fn = det_info(ground_truth_centers, predicted_centers, classes, threshold)
    return tp / (tp + fp)


def det_f1score(precision, recall, tolerance=0.00):
    return (precision * recall) / (precision + recall + tolerance)


def det_false_rate(ground_truth_centers, predicted_centers, classes, threshold=5):
    tp, fp, fn = det_info(ground_truth_centers, predicted_centers, classes, threshold)
    return 1.0 - tp / (tp + fp + fn)
