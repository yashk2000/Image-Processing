from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name = "flatten")(headModel)
        headModel = Dense(D, activation = "relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Dense(classes, activation = "softmax")(headModel)

        return headModel