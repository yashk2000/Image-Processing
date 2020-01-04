from config import ageGenderConfig as config
from computervision.nn.mxconv import MxAgeGenderNet
from computervision.utils import AgeGenderHelper
from computervision.mxcallbacks import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-p", "--prefix", required=True)
ap.add_argument("-s", "--start-epoch", type=int, default=0)
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
filename="training_{}.log".format(args["start_epoch"]), filemode="w")

batchSize = config.BATCH_SIZE * config.NUM_DEVICES
means = json.loads(open(config.DATASET_MEAN).read())

trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=7,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

opt = mx.optimizer.SGD(learning_rate=1e-3, momentum=0.9, wd=0.0005, rescale_grad=1.0 / batchSize)

checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None

if args["start_epoch"] <= 0:
    print("[INFO] building network...")
    model = MxAgeGenderNet.build(config.NUM_CLASSES)

else:
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    (model, argParams, auxParams) = mx.model.load_checkpoint(
    checkpointsPath, args["start_epoch"])

model = mx.model.FeedForward(
    ctx=[mx.gpu(2), mx.gpu(3)],
    symbol=model,
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    num_epoch=110,
    begin_epoch=args["start_epoch"]
)

batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

#TODO check to see if one-off accuracy callback should be used

print("[INFO] training network...")
model.fit(
    X=trainIter,
    eval_data=valIter,
    eval_metric=metrics,
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs
)
