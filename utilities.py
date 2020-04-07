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


def showImagesDetection(img, iter):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.savefig('./outputs/detected/img_input [' + str(iter) + '].png')
    plt.show()

def showImagesSegmentation(img, iter):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.savefig('./outputs/segmented/img_input [' + str(iter) + '].png')
    plt.show()

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
                color = color = colorMap[3]
            else:
                color = colorMap[int(imageClasses[i][j])]
            for k in range(3):
                detImage[k, i, j] = color[k]
    detImage = detImage.transpose((1, 2, 0))
    plt.imshow(detImage)
    plt.savefig('./outputs/detected/img_' + stringName + '[' + str(iter) + '].png')
    plt.show()


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
    plt.imshow(segImage)
    plt.savefig('./outputs/segmented/img_' + stringName + '[' + str(iter) + '].png')
    plt.show()


def plot_learning_curve(loss_errors, task):
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(np.array(loss_errors))
    if (task == "detection"):
        plt.savefig('./outputs/detected/curve.png')
    else:
        plt.savefig('./outputs/segmented/curve.png')

def plot_confusion_matrix(conf_matrix,operation,cmap=plt.cm.Blues):
    title = 'Normalized confusion matrix'
    cm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    print(cm)

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
