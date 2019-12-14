from computervision.nn.conv import LeNet
from computervision.nn.conv import MiniVGGNet
from computervision.nn.conv import ShallowNet
from keras.utils import plot_model

model1 = LeNet.build(28, 28, 1, 10)
plot_model(model1, to_file = "lenet.png", show_shapes = True)

model2 = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10)
plot_model(model2, to_file = "MiniVGGNet.png", show_shapes = True)

model3 = ShallowNet.build(width = 32, height = 32, depth = 3, classes = 10)
plot_model(model3, to_file = "ShallowNet.png", show_shapes = True)