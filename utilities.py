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


def showImages(img, iter):
    img = img / 2 + 0.5  # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.savefig('./outputs/detected/img_input [' + str(iter) + '].png')
    plt.show()


def showDetectedImages(image, iter, stringName):
    image[image > 0.5] = 0
    imageClasses = torch.argmax(image, 1)
    colorMap = [[255, 0, 0], [0, 0, 255], [0, 255, 0]]
    size = imageClasses.size()
    detImage = np.empty([3, size[0], size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            color = colorMap[int(imageClasses[i][j])]
            for k in range(3):
                detImage[k, i, j] = color[k]
    detImage = detImage.transpose((1, 2, 0))
    plt.imshow(detImage)
    plt.savefig('./outputs/detected/img_' + stringName + '[' + str(iter) + '].png')
    plt.show()
