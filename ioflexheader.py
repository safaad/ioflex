import optuna
import numpy as np


# OPTUNA SPECIFIC
SAMPLER_MAP = {
    "gp": ("Gaussian Processes", optuna.samplers.GPSampler),
    "rand": ("Random Search", optuna.samplers.RandomSampler),
    "nsga": ("NSGA-II", optuna.samplers.NSGAIISampler),
    "grid": ("Grid Search", optuna.samplers.GridSampler),
    "brute": ("Brute Force", optuna.samplers.BruteForceSampler),
    "tpe": ("TPE Sampler", optuna.samplers.TPESampler),
}

PRUNER_MAP = {
    "median": optuna.pruners.MedianPruner,
    "hyper": optuna.pruners.HyperbandPruner,
    "successivehalving": optuna.pruners.SuccessiveHalvingPruner,
    "nop" : optuna.pruners.NopPruner,
}

# IO CONFIGURATIONS Parameters Space
CONFIG_MAP = {
    "romio_filesystem_type": [], #["LUSTRE:", "UFS:"],
    "romio_ds_read": ["automatic", "enable", "disable"] ,
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"],
    "striping_factor": [4, 8, 16, 24, 32, 40, 48, 64, -1],
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
    "cb_nodes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "cb_config_list": ["*:1", "*:*"]
}


## Prediction Specific
CATECORY_MAP = {
    feature: {category: idx for idx, category in enumerate(values)}
    for feature, values in CONFIG_MAP.items()
}


def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean(numerator / denominator) * 100
    return smape_value