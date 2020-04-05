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


def showImages(img,iter):
    img = img / 2 + 0.5     # unnormalize
    img = img.cpu().numpy()
    img = img.transpose((1,2,0))
    plt.imshow(img)
    plt.savefig('./outputs/detected/img_input [' + str(iter) + '].png')
    plt.show()


def showDetectedImages(image,iter,stringName):
    image[image<0.5] = 0
    values,imageClasses = torch.max(image,dim=0)
    colorMap = [[255,0,0],[0,0,255],[0,255,0],[255,255,255]]
    size = imageClasses.size()
    detImage = np.empty([3,size[0],size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            if(values[i][j]==0):
                color = color = colorMap[3]
            else:
                color = colorMap[int(imageClasses[i][j])]
            for k in range(3):
                detImage[k,i,j] = color[k]
    detImage = detImage.transpose((1,2,0))
    plt.imshow(detImage)
    plt.savefig('./outputs/detected/img_'+ stringName+'[' + str(iter) + '].png')
    plt.show()

def visualiseSegmented(segmentedImage,iter,stringName):
    colorMap = [[0,0,0],[255,255,255],[0,255,0]]
    size = segmentedImage.size()
    segImage = np.empty([3,size[0],size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            color = colorMap[int(segmentedImage[i][j])]
            for k in range(3):
                segImage[k,i,j] = color[k]
    segImage = segImage.transpose((1,2,0))
    plt.imshow(segImage)
    plt.savefig('./outputs/segmented/img_'+ stringName+'[' + str(iter) + '].png')
    plt.show()

def plot_learning_curve(loss_errors,task):
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.plot(np.array(loss_errors))
  if(task == "detection"):
      plt.savefig('./outputs/detected/curve.png')
  else:
      plt.savefig('./outputs/segmented/curve.png')
    
      