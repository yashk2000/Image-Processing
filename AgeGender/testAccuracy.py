from computervision.utils import AgeGenderHelper

from config import ageGenderConfig as config
from computervision.mxcallbacks import _compute_one_off
import mxnet as mx
import argparse
import pickle
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-p", "--prefix", required=True)
ap.add_argument("-e", "--epoch", type=int, required=True)
args = vars(ap.parse_args())

means = json.loads(open(config.DATASET_MEAN).read())

testIter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=config.BATCH_SIZE,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

print("[INFO] loading model")
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath, args["epoch"])

model = mx.model.FeedForward(
ctx=[mx.gpu(0)],
symbol=model.symbol,
arg_params=model.arg_params,
aux_params=model.aux_params)

print("[INFO] predicting on ’{}’ test data...".format(config.DATASET_TYPE))
metrics = [mx.metric.Accuracy()]
acc = model.score(testIter, eval_metric=metrics)

print("[INFO] rank-1: {:.2f}%".format(acc[0] * 100))

if config.DATASET_TYPE == "age":
    arg = model.arg_params
    aux = model.aux_params
    model = mx.mod.Module(symbol=model.symbol, context=[mx.gpu(1)])
    model.bind(data_shapes=testIter.provide_data,
    label_shapes=testIter.provide_label)
    model.set_params(arg, aux)

    le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
    agh = AgeGenderHelper(config)
    oneOff = agh.buildOneOffMappings(le)

    acc = _compute_one_off(model, testIter, oneOff)
    print("[INFO] one-off: {:.2f}%".format(acc * 100))

