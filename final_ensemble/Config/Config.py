from enum import Enum

class Config(Enum):
    EFF_NET_B2_WEIGHT_PATH = './testWeights/efficientnet_b2/43_1.30.h5'
    VIT_WEIGHT_PATH = './testWeights/vit/108_2.18.h5'
    DATA_PATH = './data/feather-in-focus'
    PREDICTION_BASE_PATH = './predictions'
    PREDICTION_FILE_POSTFIX = 'ensemble_predictions.csv'
    VIT_EXPERIMENT_BASE_PATH = './experiments/vit'
    EFF_NET_EXPERIMENT_BASE_PATH = './experiments/efficientnet_b2'
    