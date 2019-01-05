import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import normalize

# src/neural/trainingData/testHdn/7.png

def lettreGenerator():
    iniLettre = 65
    for i in range(26):
        yield chr(iniLettre + i)

def predict(model):
    im = cv2.imread("src/neural/trainingData/testHuman/A.png", 0)
    return predict_lettre(model, im)

def predict_lettre(model, img):
    gene = lettreGenerator()
    data = cv2.resize(img, (28,28))
    data = cv2.threshold(data,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    data = normalize(data, axis=1)
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()
    fig, ax = plt.subplots()
    data = np.reshape(data, (28,28,1))
    # data = data.flatten()
    print(data.shape)
    pred = model.predict([[data]])[0]
    labels = list(gene)
    proba = ax.bar(np.arange(26), pred,
                alpha=0.4, color='r',
                label='Prédictions')

    ax.set_xticks(np.arange(26))
    ax.set_xticklabels(labels)
    print("argmax : ", pred)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return labels[pred.argmax()]
