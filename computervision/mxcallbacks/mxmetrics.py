import mxnet as mx
import logging

def one_off_callback(trainIter, testIter, oneOff, ctx):
    def _callback(iterNum, sym, arg, aux):
        model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(data_shapes=testIter.provide_data, label_shapes=testIter.provide_label)
        model.set_params(arg, aux)

        trainMAE = _compute_one_off(model, trainIter, oneOff)
        testMAE = _compute_one_off(model, testIter, oneOff)
        logging.info("Epoch[{}] Train-one-off={:.5f}".format(iterNum, trainMAE))
        logging.info("Epoch[{}] Test-one-off={:.5f}".format(iterNum, testMAE))

    return _callback

def _compute_one_off(model, dataIter, oneOff):

    total = 0
    correct = 0

    for (preds, _, batch) in model.iter_predict(dataIter):

        predictions = preds[0].asnumpy().argmax(axis=1)
        labels = batch.label[0].asnumpy().astype("int")

        for (pred, label) in zip(predictions, labels):

            if label in oneOff[pred]:
                correct += 1

            total += 1

    return correct / float(total)

