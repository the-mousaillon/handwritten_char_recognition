import src.neural.RNN.neuralBuilderV2 as neurone
import src.neural.RNN.neuralBuilderConvNet as convNN
import src.neural.RNN.predictTools as predictTools
import src.neural.dataBuilder.buildKaggleV2 as bdd
import keras

if __name__ == "__main__":
    # model = neurone.buildModel()

    model = keras.models.load_model("src/neural/RNN/model/hand_written_recognition_Conv.model")