import numpy as np
import cv2


def segmentationAccuracy(segmented, targets, classes):
    accuracies = [0, 0, 0, 0]
    avg_acc = 0
    targets = targets.cpu()
    segmented = segmented.cpu()
    for c in classes:
        true_positive = np.logical_and((targets == c), (segmented == c)).sum()
        true_negative = np.logical_and((targets != c),(segmented!=c)).sum()
        false_positive = np.logical_and((targets == c),(segmented !=c )).sum()
        false_negative = np.logical_and((targets != c),(segmented ==c )).sum()
        accuracies[c] = (true_positive.item() + true_negative.item()) / (true_positive.item()+true_negative.item()+false_positive.item()+false_negative.item())  
        avg_acc = avg_acc + accuracies[c]
    avg_acc = avg_acc / len(classes)
    accuracies[3] = avg_acc
    return accuracies


def seg_iou(ground_truth, predicted, classes):
    avg_iou = 0
    iou = [0, 0, 0, 0]
    for c in classes:
        gt = (ground_truth.cpu() == c)
        pr = (predicted.cpu() == c)
        intersection = np.logical_and(gt, pr)
        union = np.logical_or(gt, pr)
        iou_score = intersection.sum().item() / union.sum().item()
        iou[c] = iou_score
        avg_iou += iou_score
    iou[3] = avg_iou / len(classes)
    return iou


def getArea(contour):
    contour = contour.squeeze()
    return ((contour.max(axis=0) - contour.min(axis=0)).prod())


def get_centers(im, classes, thres=25):
    kernel = np.ones((3, 3), np.uint8)
    border = 20
    im = im.astype('uint8')
    im = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours = []
    for c in classes:
        c_im = np.zeros((closing.shape)) + 255
        c_im[np.all(closing == c, axis=-1)] = c
        c_im = c_im.astype('uint8')
        imgray = cv2.cvtColor(c_im, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 150, 255, 0)

        i, c_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(c_contours[1:])

    test = np.zeros((closing.shape)) + 255
    cv2.drawContours(test, contours, -1, (0, 0, 0), 1)

    centers = []
    colors = []
    no_points = []

    for contour in contours:

        if cv2.contourArea(contour) < thres:
            continue
        mean = (np.mean(contour, axis=0, dtype=np.int32).squeeze())

        color = closing[mean[1], mean[0]]
        mean = mean - border
        centers.append([mean[1], mean[0]])
        colors.append(color)
        no_points.append(len(contour))

    centers = np.array(centers)
    colors = np.array(colors)
    no_points = np.array(no_points)
    indicies = no_points.argsort()
    centers = centers[indicies][::-1]
    colors = colors[indicies][::-1]

    return [centers, colors]


def det_info(ground_truth_centers, predicted_centers, classes, threshold=10.0):
    tp = np.zeros((len(classes)))
    fp = np.zeros((len(classes)))
    fn = np.zeros((len(classes)))

    for j, c in enumerate(classes):
        gts_indicies = np.all(ground_truth_centers[1] == c, axis=-1)
        gts_centers = ground_truth_centers[0][gts_indicies]

        prs_indicies = np.all(predicted_centers[1] == c, axis=-1)
        prs_centers = predicted_centers[0][prs_indicies]

        correct_indices = []

        from scipy.spatial import KDTree
        if not (gts_centers.size == 0 or prs_centers.size == 0):
            tree = KDTree(gts_centers)
            neighbor_dists, neighbor_indices = tree.query(prs_centers)
            sorted_indices = (np.lexsort((neighbor_dists, neighbor_indices)))
            correct_indices = [sorted_indices[i] for i in
                               range(len(sorted_indices)) if
                               (neighbor_indices[sorted_indices[i]] != neighbor_indices[sorted_indices[i - 1]]
                                or i == 0) and neighbor_dists[sorted_indices[i]] < threshold]

        tp[j] = (len(correct_indices))
        fp[j] = (len(prs_centers) - len(correct_indices))
        fn[j] = (len(gts_centers) - len(correct_indices))

    return tp, fp, fn


def det_metrics(ground_truth_centers, predicted, classes, threshold=10, tolerance=0.001):
    batch_tp = np.zeros((len(classes)))
    batch_fp = np.zeros((len(classes)))
    batch_fn = np.zeros((len(classes)))

    for i in range(len(predicted)):
        predicted_centers = get_centers(predicted[i], classes)
        tp, fp, fn = det_info(ground_truth_centers[i], predicted_centers, classes, threshold)
        batch_tp += tp
        batch_fp += fp
        batch_fn += fn

    accuracy = batch_tp / (batch_tp + batch_fp + batch_fn + tolerance)
    recall = batch_tp / (batch_tp + batch_fn + tolerance)
    precision = batch_tp / (batch_tp + batch_fp + tolerance)
    f1score = 2 * (precision * recall) / (precision + recall + tolerance)
    false_rate = batch_fp + batch_fn / (batch_tp + batch_fp + batch_fn + tolerance)
    return accuracy, recall, precision, f1score, false_rate


def det_confusion_matrix(ground_truth_centers, predicted, classes, threshold=10):
    from sklearn.metrics import confusion_matrix

    from scipy.spatial import KDTree
    confusion = np.zeros((len(classes) + 1, len(classes) + 1))

    for i in range(len(predicted)):
        predicted_centers = get_centers(predicted[i], classes)
        y_pred = np.ones(len(ground_truth_centers[i][1])) * 3
        if len(predicted_centers[0]) != 0:
            tree = KDTree(ground_truth_centers[i][0])
            neighbor_dists, neighbor_indices = tree.query(predicted_centers[0])
            y_pred[neighbor_indices] = predicted_centers[1].argmax(axis=1)

        y_true = ground_truth_centers[i][1].argmax(axis=1)
        confusion += confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return confusion

def seg_confusion_matrix(ground_truth,segmented,classes):
    from sklearn.metrics import confusion_matrix
    
    confusion = np.zeros((len(classes), len(classes)))
    y_true = ground_truth.view(-1)
    y_pred = segmented.view(-1)
    confusion = confusion_matrix(y_true, y_pred, labels=classes)
    return confusion