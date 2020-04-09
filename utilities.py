import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def setSeed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getDev():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        avDev = torch.device("cuda")
    else:
        avDev = torch.device("cpu")
    return avDev


def reduce_brightness(img, value=60):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v[v < value] = 0
    v[v >= value] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def plot_det_seg_visuals(ims, det, seg, name='table.png'):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(3, len(ims))
    # axarr[0, 0].set_title('Images')
    # axarr[0, 1].set_title('Ground Truths')
    # axarr[0, 2].set_title('Predictions')

    for i in range(len(ims)):
        axarr[0, i].imshow(ims[i])
        axarr[1, i].imshow(det[i])
        axarr[2, i].imshow(seg[i])

        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[2, i].axis('off')

    f.tight_layout()
    plt.savefig('./outputs/' + name)
    plt.show()
def plot_det_seg_visualsInference(ims, det, seg,folder, name='table.png'):
    import matplotlib.pyplot as plt
    plt.close()
    f, axarr = plt.subplots(3, len(ims))

    for i in range(len(ims)):
        axarr[0, i].imshow(ims[i])
        axarr[1, i].imshow(det[i])
        axarr[2, i].imshow(seg[i])

        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        axarr[2, i].axis('off')

    f.tight_layout()
    plt.savefig(folder+'/' + name)
    plt.show()


def plot_visuals(ims, gts, prs, name='table.png'):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(len(ims), 3)

    for i in range(len(ims)):
        axarr[i, 0].imshow(ims[i])
        axarr[i, 1].imshow(gts[i])
        axarr[i, 2].imshow(prs[i])

        axarr[i, 0].axis('off')
        axarr[i, 1].axis('off')
        axarr[i, 2].axis('off')

    plt.savefig('./outputs/' + name)
    plt.show()


def showImagesDetection(img, iter):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig('./outputs/detected/img_' + '[' + str(iter) + ']' + '_input.png')
    # plt.show()


def showImagesSegmentation(img, iter):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig('./outputs/segmented/img_' + '[' + str(iter) + ']' + '_input.png')
    # plt.show()
def saveImagesInference(img, folder):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(img)
    return img


def get_predected_centers(images):
    colorMap = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]]
    batch_centers = []
    for image in images:
        image[image < 0.5] = 0
        values, imageClasses = torch.max(image, dim=0)
        colorMap = np.array(colorMap)
        imageClasses = np.array(imageClasses)

        indices = (values != 0)
        centers = np.array(np.argwhere(indices).T)
        colors = colorMap[imageClasses[indices]]
        batch_centers.append([centers, colors])

    return batch_centers


def get_colored_image(images):
    detImages = []
    for image in images:
        image[image < 0.5] = 0
        values, imageClasses = torch.max(image, dim=0)
        colorMap = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]]
        size = imageClasses.size()
        detImage = np.empty([3, size[0], size[1]])
        for i in range(size[0]):
            for j in range(size[1]):
                if (values[i][j] == 0):
                    color = colorMap[3]
                else:
                    color = colorMap[int(imageClasses[i][j])]
                for k in range(3):
                    detImage[k, i, j] = color[k]
        detImage = detImage.transpose((1, 2, 0))
        detImages.append(detImage)
    return np.array(detImages)


def showDetectedImages(image, iter, stringName):
    image[image < 0.5] = 0
    values, imageClasses = torch.max(image, dim=0)
    colorMap = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]]
    size = imageClasses.size()
    detImage = np.empty([3, size[0], size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            if (values[i][j] == 0):
                color = colorMap[3]
            else:
                color = colorMap[int(imageClasses[i][j])]
            for k in range(3):
                detImage[k, i, j] = color[k]
    detImage = detImage.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(detImage)
    plt.savefig('./outputs/detected/img_' + '[' + str(iter) + ']_' + stringName + '.png')
    # plt.show()

def saveDetectedInference(image,folder):
    image[image < 0.5] = 0
    values, imageClasses = torch.max(image, dim=0)
    colorMap = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]]
    size = imageClasses.size()
    detImage = np.empty([3, size[0], size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            if (values[i][j] == 0):
                color = color = colorMap[3]
            else:
                color = colorMap[int(imageClasses[i][j])]
            for k in range(3):
                detImage[k, i, j] = color[k]
    detImage = detImage.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(detImage)
    return detImage


def visualiseSegmented(segmentedImage, iter, stringName):
    colorMap = [[0, 0, 0], [255, 255, 255], [0, 255, 0]]
    size = segmentedImage.size()
    segImage = np.empty([3, size[0], size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            color = colorMap[int(segmentedImage[i][j])]
            for k in range(3):
                segImage[k, i, j] = color[k]
    segImage = segImage.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(segImage)
    plt.savefig('./outputs/segmented/img_' + '[' + str(iter) + ']_' + stringName + '.png')
    plt.show()

def saveSegmentedInference(image,folder):
    segmentedImage = torch.argmax(image,dim=0)
    colorMap = [[0, 0, 0], [255, 255, 255], [0, 255, 0]]
    size = segmentedImage.size()
    segImage = np.empty([3, size[0], size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            color = colorMap[int(segmentedImage[i][j])]
            for k in range(3):
                segImage[k, i, j] = color[k]
    segImage = segImage.transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(segImage)
    return segImage

def plot_learning_curve(loss_errors, task):
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(np.array(loss_errors))
    if (task == "detection"):
        plt.savefig('./outputs/detected/curve.png')
    else:
        plt.savefig('./outputs/segmented/curve.png')


def plot_confusion_matrix(conf_matrix, operation, cmap=plt.cm.Blues, tolerance=0.0001):
    title = 'Normalized confusion matrix'
    cm = conf_matrix / (conf_matrix.sum() + tolerance)


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if operation == "segmentation":
        plt.savefig('./outputs/segmented/confusion_matrix.png')
    else:
        plt.savefig('./outputs/detected/confusion_matrix.png')
