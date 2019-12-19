from computervision.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True)
ap.add_argument("-m", "--model", required = True)
args = vars(ap.parse_args())

print("[INFO] Loading model")
model = pickle.load(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] Predicting")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

print("[INFO] Rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] Rank-5: {:.2f}%".format(rank5 * 100))

db.close()