# import src.neural.dataBuilder.build as buildTool
# import src.neural.RNN.neuralBuilder as buildRNN
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import src.neural.RNN.predictTools as predTools
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Building the dataset ....")
    #buildTool.makeData()
    print("Loading the model")
    model = load_model("src/neural/RNN/model/hand_written_recognition.model")
    img = cv2.imread("src/neural/trainingData/testHdn/8.png",0)
    plt.imshow(img)
    plt.show()
    print("prediction ...")
    p = predTools.predict_lettre(model, img)
    print(p)
