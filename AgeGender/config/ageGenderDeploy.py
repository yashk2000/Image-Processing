from age_gender_config import OUTPUT_BASE
from os import path

class AgeGenderDeploy:

    DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
    AGE_NETWORK_PATH = "checkpoints/age"
    AGE_PREFIX = "agenet"
    AGE_EPOCH = 150
    AGE_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "age_le.cpickle"])
    AGE_MEANS = path.sep.join([OUTPUT_BASE, "age_adience_mean.json"])

    GENDER_NETWORK_PATH = "checkpoints/gender"
    GENDER_PREFIX = "gendernet"
    GENDER_EPOCH = 110
    GENDER_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "gender_le.cpickle"])
    GENDER_MEANS = path.sep.join([OUTPUT_BASE, "gender_adience_mean.json"])