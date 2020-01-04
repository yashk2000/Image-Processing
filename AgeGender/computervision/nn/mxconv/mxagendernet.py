import mxnet as mx

class MxAgeGenderNet:

    @staticmethod
    def build(classes):
        data = mx.sym.Variable("data")

        conv1_1 = mx.sym.Convolution(data=data, kernel=(7, 7),
        stride=(4, 4), num_filter=96)
        act1_1 = mx.sym.Activation(data=conv1_1, act_type="relu")
        bn1_1 = mx.sym.BatchNorm(data=act1_1)
        pool1 = mx.sym.Pooling(data=bn1_1, pool_type="max",
        kernel=(3, 3), stride=(2, 2))
        do1 = mx.sym.Dropout(data=pool1, p=0.25)

        conv2_1 = mx.sym.Convolution(data=do1, kernel=(5, 5),
        pad=(2, 2), num_filter=256)
        act2_1 = mx.sym.Activation(data=conv2_1, act_type="relu")
        bn2_1 = mx.sym.BatchNorm(data=act2_1)
        pool2 = mx.sym.Pooling(data=bn2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        do2 = mx.sym.Dropout(data=pool2, p=0.25)

        conv2_1 = mx.sym.Convolution(data=do2, kernel=(3, 3),
        pad=(1, 1), num_filter=384)
        act2_1 = mx.sym.Activation(data=conv2_1, act_type="relu")
        bn2_1 = mx.sym.BatchNorm(data=act2_1)
        pool2 = mx.sym.Pooling(data=bn2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
        do3 = mx.sym.Dropout(data=pool2, p=0.25)

        flatten = mx.sym.Flatten(data=do3)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=512)
        act4_1 = mx.sym.Activation(data=fc1, act_type="relu")
        bn4_1 = mx.sym.BatchNorm(data=act4_1)
        do4 = mx.sym.Dropout(data=bn4_1, p=0.5)

        fc2 = mx.sym.FullyConnected(data=do4, num_hidden=512)
        act5_1 = mx.sym.Activation(data=fc2, act_type="relu")
        bn5_1 = mx.sym.BatchNorm(data=act5_1)
        do5 = mx.sym.Dropout(data=bn5_1, p=0.5)

        fc3 = mx.sym.FullyConnected(data=do5, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc3, name="softmax")

        return model