import mxnet as mx

class MxAlexNet:
	@staticmethod
	def build(classes):
		data = mx.sym.Variable("data")

		conv1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
		act1_1 = mx.sym.LeakyReLU(data=conv1_1, act_type="elu")
		bn1_1 = mx.sym.BatchNorm(data=act1_1)
		pool1 = mx.sym.Pooling(data=bn1_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
		do1 = mx.sym.Dropout(data=pool1, p=0.25)

		conv2_1 = mx.sym.Convolution(data=do1, kernel=(5, 5), pad=(2, 2), num_filter=256)
		act2_1 = mx.sym.LeakyReLU(data=conv2_1, act_type="elu")
		bn2_1 = mx.sym.BatchNorm(data=act2_1)
		pool2 = mx.sym.Pooling(data=bn2_1, pool_type="max", kernel=(3, 3), stride=(2, 2))
		do2 = mx.sym.Dropout(data=pool2, p=0.25)

		conv3_1 = mx.sym.Convolution(data=do2, kernel=(3, 3), pad=(1, 1), num_filter=384)
		act3_1 = mx.sym.LeakyReLU(data=conv3_1, act_type="elu")
		bn3_1 = mx.sym.BatchNorm(data=act3_1)
		conv3_2 = mx.sym.Convolution(data=bn3_1, kernel=(3, 3), pad=(1, 1), num_filter=384)
		act3_2 = mx.sym.LeakyReLU(data=conv3_2, act_type="elu")
		bn3_2 = mx.sym.BatchNorm(data=act3_2)
		conv3_3 = mx.sym.Convolution(data=bn3_2, kernel=(3, 3), pad=(1, 1), num_filter=256)
		act3_3 = mx.sym.LeakyReLU(data=conv3_3, act_type="elu")
		bn3_3 = mx.sym.BatchNorm(data=act3_3)
		pool3 = mx.sym.Pooling(data=bn3_3, pool_type="max", kernel=(3, 3), stride=(2, 2))
		do3 = mx.sym.Dropout(data=pool3, p=0.25)

		flatten = mx.sym.Flatten(data=do3)
		fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
		act4_1 = mx.sym.LeakyReLU(data=fc1, act_type="elu")
		bn4_1 = mx.sym.BatchNorm(data=act4_1)
		do4 = mx.sym.Dropout(data=bn4_1, p=0.5)

		fc2 = mx.sym.FullyConnected(data=do4, num_hidden=4096)
		act5_1 = mx.sym.LeakyReLU(data=fc2, act_type="elu")
		bn5_1 = mx.sym.BatchNorm(data=act5_1)
		do5 = mx.sym.Dropout(data=bn5_1, p=0.5)

		fc3 = mx.sym.FullyConnected(data=do5, num_hidden=classes)
		model = mx.sym.SoftmaxOutput(data=fc3, name="softmax")

		return model