import mxnet as mx

class MxResNet:
	@staticmethod
	def residual_module(data, K, stride, red=False, bnEps=2e-5, bnMom=0.9):
		shortcut = data

		bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bnEps, momentum=bnMom)
		act1 = mx.sym.Activation(data=bn1, act_type="relu")
		conv1 = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=int(K * 0.25), no_bias=True)

		bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps, momentum=bnMom)
		act2 = mx.sym.Activation(data=bn2, act_type="relu")
		conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=stride, num_filter=int(K * 0.25), no_bias=True)

		bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bnEps, momentum=bnMom)
		act3 = mx.sym.Activation(data=bn3, act_type="relu")
		conv3 = mx.sym.Convolution(data=act3, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=K, no_bias=True)

		if red:
			shortcut = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=stride, num_filter=K, no_bias=True)

		add = conv3 + shortcut

		return add

	@staticmethod
	def build(classes, stages, filters, bnEps=2e-5, bnMom=0.9):
		data = mx.sym.Variable("data")

		bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom)
		conv1_1 = mx.sym.Convolution(data=bn1_1, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=filters[0], no_bias=True)
		bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=bnEps, momentum=bnMom)
		act1_2 = mx.sym.Activation(data=bn1_2, act_type="relu")
		pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
		body = pool1

		for i in range(0, len(stages)):
			stride = (1, 1) if i == 0 else (2, 2)
			body = MxResNet.residual_module(body, filters[i + 1], stride, red=True, bnEps=bnEps, bnMom=bnMom)

			for j in range(0, stages[i] - 1):
				body = MxResNet.residual_module(body, filters[i + 1], (1, 1), bnEps=bnEps, bnMom=bnMom)

		bn2_1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=bnEps, momentum=bnMom)
		act2_1 = mx.sym.Activation(data=bn2_1, act_type="relu")
		pool2 = mx.sym.Pooling(data=act2_1, pool_type="avg", global_pool=True, kernel=(7, 7))

		flatten = mx.sym.Flatten(data=pool2)
		fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
		model = mx.sym.SoftmaxOutput(data=fc1, name="softmax")

		return model

if __name__ == "__main__":
	model = MxResNet.build(1000, (3, 4, 6, 3), (64, 256, 512, 1024, 2048))
	v = mx.viz.plot_network(model, shape={"data": (1, 3, 224, 224)}, node_attrs={"shape": "rect", "fixedsize": "false"})
	v.render()