from keras import backend as K
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization


class MiniGoogLeNet:

    @staticmethod
    def _conv_module(x:'keras.layers.convolutional' , K:int, ksize:tuple, stride:tuple, chanDim:int, padding:str = "same"):
        (kX, kY) = ksize
        x = Conv2D(K, (kX, kY), strides = stride, padding = padding)(x)
        x = BatchNormalization(axis = chanDim)(x)
        x = Activation("relu")(x)

        # return the block
        return x
    

    @staticmethod
    def _inception_module(x:'keras.layers.convolutional', numK1x1:int, numK3x3:int, chanDim:int):
        conv_1x1 = MiniGoogLeNet._conv_module(x, numK1x1, (1, 1), (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet._conv_module(x, numK3x3, (3, 3), (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis = chanDim)

        return x
    
    @staticmethod
    def _downsample_module(x, K, chanDim):
        conv_3x3 = MiniGoogLeNet._conv_module(x, K, (3, 3), (2, 2), chanDim, padding = "valid")
        pool = MaxPooling2D((3, 3), strides = (2, 2))(x)
        x = concatenate([conv_3x3, pool], axis = chanDim)

        return x
    
    @staticmethod
    def build(width:int , height:int, depth:int, classes:int):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        inputs = Input(shape = inputShape)
        x = MiniGoogLeNet._conv_module(inputs, 96, (3, 3), (1, 1), chanDim)

        x = MiniGoogLeNet._inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet._inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet._downsample_module(x, 80, chanDim)

        x = MiniGoogLeNet._inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet._inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet._inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet._inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet._downsample_module(x, 96, chanDim)

        x = MiniGoogLeNet._inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet._inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name = "minigooglenet")

        return model