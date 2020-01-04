import mxnet as mx

class MxVGGNet:
	@staticmethod
	def build(classes):
		data = mx.sym.Variable("data")

		conv1_1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
		act1_1 = mx.sym.LeakyReLU(data=conv1_1, act_type="prelu", name="act1_1")
		bn1_1 = mx.sym.BatchNorm(data=act1_1, name="bn1_1")
		conv1_2 = mx.sym.Convolution(data=bn1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
		act1_2 = mx.sym.LeakyReLU(data=conv1_2, act_type="prelu", name="act1_2")
		bn1_2 = mx.sym.BatchNorm(data=act1_2, name="bn1_2")
		pool1 = mx.sym.Pooling(data=bn1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
		do1 = mx.sym.Dropout(data=pool1, p=0.25)

		conv2_1 = mx.sym.Convolution(data=do1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
		act2_1 = mx.sym.LeakyReLU(data=conv2_1, act_type="prelu", name="act2_1")
		bn2_1 = mx.sym.BatchNorm(data=act2_1, name="bn2_1")
		conv2_2 = mx.sym.Convolution(data=bn2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
		act2_2 = mx.sym.LeakyReLU(data=conv2_2, act_type="prelu", name="act2_2")
		bn2_2 = mx.sym.BatchNorm(data=act2_2, name="bn2_2")
		pool2 = mx.sym.Pooling(data=bn2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
		do2 = mx.sym.Dropout(data=pool2, p=0.25)

		conv3_1 = mx.sym.Convolution(data=do2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
		act3_1 = mx.sym.LeakyReLU(data=conv3_1, act_type="prelu", name="act3_1")
		bn3_1 = mx.sym.BatchNorm(data=act3_1, name="bn3_1")
		conv3_2 = mx.sym.Convolution(data=bn3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
		act3_2 = mx.sym.LeakyReLU(data=conv3_2, act_type="prelu", name="act3_2")
		bn3_2 = mx.sym.BatchNorm(data=act3_2, name="bn3_2")
		conv3_3 = mx.sym.Convolution(data=bn3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
		act3_3 = mx.sym.LeakyReLU(data=conv3_3, act_type="prelu", name="act3_3")
		bn3_3 = mx.sym.BatchNorm(data=act3_3, name="bn3_3")
		pool3 = mx.sym.Pooling(data=bn3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
		do3 = mx.sym.Dropout(data=pool3, p=0.25)

		conv4_1 = mx.sym.Convolution(data=do3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
		act4_1 = mx.sym.LeakyReLU(data=conv4_1, act_type="prelu", name="act4_1")
		bn4_1 = mx.sym.BatchNorm(data=act4_1, name="bn4_1")
		conv4_2 = mx.sym.Convolution(data=bn4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
		act4_2 = mx.sym.LeakyReLU(data=conv4_2, act_type="prelu", name="act4_2")
		bn4_2 = mx.sym.BatchNorm(data=act4_2, name="bn4_2")
		conv4_3 = mx.sym.Convolution(data=bn4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
		act4_3 = mx.sym.LeakyReLU(data=conv4_3, act_type="prelu", name="act4_3")
		bn4_3 = mx.sym.BatchNorm(data=act4_3, name="bn4_3")
		pool4 = mx.sym.Pooling(data=bn4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
		do4 = mx.sym.Dropout(data=pool4, p=0.25)

		conv5_1 = mx.sym.Convolution(data=do4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
		act5_1 = mx.sym.LeakyReLU(data=conv5_1, act_type="prelu", name="act5_1")
		bn5_1 = mx.sym.BatchNorm(data=act5_1, name="bn5_1")
		conv5_2 = mx.sym.Convolution(data=bn5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
		act5_2 = mx.sym.LeakyReLU(data=conv5_2, act_type="prelu", name="act5_2")
		bn5_2 = mx.sym.BatchNorm(data=act5_2, name="bn5_2")
		conv5_3 = mx.sym.Convolution(data=bn5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
		act5_3 = mx.sym.LeakyReLU(data=conv5_3, act_type="prelu", name="act5_3")
		bn5_3 = mx.sym.BatchNorm(data=act5_3, name="bn5_3")
		pool5 = mx.sym.Pooling(data=bn5_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool5")
		do5 = mx.sym.Dropout(data=pool5, p=0.25)

		flatten = mx.sym.Flatten(data=do5, name="flatten")
		fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096, name="fc1")
		act6_1 = mx.sym.LeakyReLU(data=fc1, act_type="prelu", name="act6_1")
		bn6_1 = mx.sym.BatchNorm(data=act6_1, name="bn6_1")
		do6 = mx.sym.Dropout(data=bn6_1, p=0.5)

		fc2 = mx.sym.FullyConnected(data=do6, num_hidden=4096, name="fc2")
		act7_1 = mx.sym.LeakyReLU(data=fc2, act_type="prelu", name="act7_1")
		bn7_1 = mx.sym.BatchNorm(data=act7_1, name="bn7_1")
		do7 = mx.sym.Dropout(data=bn7_1, p=0.5)

		fc3 = mx.sym.FullyConnected(data=do7, num_hidden=classes, name="fc3")
		model = mx.sym.SoftmaxOutput(data=fc3, name="softmax")

		return model