from keras.applications import VGG16
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type = int, default = 1)
args = vars(ap.parse_args())

print("[INFO] Loading")
model = VGG16(weights = "imagenet", include_top = args["include_top"] > 0)
print("[INFO] Shwoing layers")

for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
