import cv2
import keras
import matplotlib.pyplot as plt


def extractKernel(nbParCouche, model):
    kernelList=[]
    for layer in model.layers:
        if str(layer).find("Conv2D") != -1:
            for i in range(nbParCouche):
                kernelList += [layer.get_weights()[0][:,:,0,i]]
    return kernelList


def convolve(im, kernel):
    return cv2.filter2D(im,-1,kernel)


def plotKernels(nbParCouche, model):
    kernelList = extractKernel(nbParCouche, model)
    columns = nbParCouche
    rows = 5
    fig=plt.figure(figsize=(10, 10))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(kernelList[i-1], cmap=plt.cm.binary)
    plt.show()

def plotConvolve(im, nbParCouche, model):
    kernelList = extractKernel(nbParCouche, model)
    columns = nbParCouche
    rows = 5
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(convolve(im, kernelList[i-1]), cmap=plt.cm.binary)
    plt.show()