from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.regularizers import l2
from keras import backend as K

class DeeperGoogLeNet:

    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding = "same", reg = 0.0005, name = None):
        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"
        
        x = Conv2D(K, (kX, kY), strides = stride, padding = padding, kernel_regularizer = l2(reg), name = convName)(x)
        x = BatchNormalization(axis = chanDim, name = bnName)(x)
        x = Activation("relu", name = actName)(x)


        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg = 0.0005):
        first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1, (1, 1), chanDim, reg = reg, name = stage + "_first")

        second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1), chanDim, reg = reg, name = stage + "_second1")
        second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3, (1, 1), chanDim, reg = reg, name = stage + "_second2")
        
        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1, (1, 1), chanDim, reg = reg, name = stage + "_third1")
        third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5, (1, 1), chanDim, reg = reg, name = stage + "_third2")

        fourth = MaxPooling2D((3, 3), strides = (1, 1), padding = "same", name = stage + "_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj, 1, 1, (1, 1), chanDim, reg = reg, name = stage + "_fourth")

        x = concatenate([first, second, third, fourth], axis = chanDim, name = stage + "_mixed")

        return x

    @staticmethod
    def build(width, height, depth, classes, reg = 0.0005):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channles_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape = inputShape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chanDim, reg = reg, name = "block1")
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = "same", name = "pool2")
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg = reg, name = "block3")
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = "same", name = "pool2")(x)

        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg = reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg = reg)
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = "same", name = "pool3")(x)

        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg = reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg = reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg = reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4e", reg = reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg = reg)
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = "same", name = "pool4")(x)

        x = AveragePooling2D((4, 4), name = "pool5")(x)
        x = Dropout(0.4, name = "do")(x)

        x = Flatten(name = "flstten")(x)
        x = Dense(classes, kernel_regularizer = l2(reg), name = "labels")(x)
        x = Activation("softmax", name = "softmax")(x)

        model = Model(inputs, x, name = "googlenet")

        return model