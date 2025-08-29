import os
import optuna
import nevergrad as ng
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

# NNEVERGRAD SPECIFIC
OPTIMIZER_MAP = {
    "ngioh": ("NgIohTuned", ng.optimizers.NgIoh),
    "twopde": ("Two Points Differential Evolution", ng.optimizers.TwoPointsDE),
    "pdopo": ("PortfolioDiscreteOnePlusOne",  ng.optimizers.PortfolioDiscreteOnePlusOne),
    "tbpsa": ("TBPSA Test-based population-size adaptation", ng.optimizers.TBPSA),
    "ngopt": ("NGOpt", ng.optimizers.NGOpt)
}

# IO CONFIGURATIONS Parameters Space
CONFIG_ROMIO_MAP = {
    "romio_filesystem_type": [], #["LUSTRE:", "UFS:"],
    "romio_ds_read": ["automatic", "enable", "disable"] ,
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"],
    "striping_factor": [4, 8, 16, 24, 32, 40, 48, 64, -1],
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
    "cb_nodes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "cb_config_list": ["*:1", "*:*"],
    "romio_no_indep_rw": ["true", "false"]
}

CONFIG_CRAY_MAP = {
    "romio_filesystem_type": [],
    "romio_ds_read": ["automatic", "enable", "disable"] ,
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"],
    "striping_factor": [4, 8, 16, 24, 32, 40, 48, 64, -1],
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
    ## doesn't make in sense in cray-mpich as it's  multiplier*striping_factor
    # "cb_nodes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "cb_config_list": ["*:1", "*:*"],
    "cray_cb_nodes_multiplier": [1, 2],
    "cray_cb_write_lock_mode": [0, 1, 2],
    "direct_io": ["true", "false"],
    "aggregator_placement_stride": [],
    "romio_no_indep_rw": ["true", "false"]
}

CONFIG_OMPIO_MAP = {
    # TODO
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
}

def get_config_map(hints):
    CONFIG_MAP = {
        "cray": CONFIG_CRAY_MAP,
        "ompio": CONFIG_OMPIO_MAP,
        "romio": CONFIG_ROMIO_MAP,
    }.get(
        hints, CONFIG_ROMIO_MAP
    )  # Use ROMIO as default fallback
    return CONFIG_MAP

## Prediction Specific
def build_category_map(config_map):
    return {
        feature: {category: idx for idx, category in enumerate(values)}
        for feature, values in config_map.items()
    }


def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean(numerator / denominator) * 100
    return smape_value

def set_hints_with_ioflex(config_dict, config_path):
    
    with open(config_path, "w") as config_file:
        separator = " = "
        # Special handling of some keys
        for key, val in config_dict.items():
            match key:
                case "striping_unit":
                    config_file.write(f"cb_buffer_size{separator}{val}\n")
                case "romio_filesystem_type":
                    os.environ.update({"ROMIO_FSTYPE_FORCE": val})
                case "cb_config_list":
                    config_file.write(f'{key}{separator}"{val}"\n')
                    continue
            config_file.write(f"{key}{separator}{val}\n")

def set_hints_env_romio(config_dict, config_path):
    with open(config_path, "w") as config_file:
        separator = " "
        # Special handling of some keys
        for key, val in config_dict.items():
            match key:
                case "striping_unit":
                    config_file.write(f"cb_buffer_size{separator}{val}\n")
                case "romio_filesystem_type":
                    os.environ.update({"ROMIO_FSTYPE_FORCE": val})
                    
            config_file.write(f"{key}{separator}{val}\n")
            
def set_hints_env_cray(config_dict, config_path):
    
    # For all files to be adjusted
    crayhints = ["*"]
    separator = "="
    
    for key, val in config_dict.items():
        if key == "cb_config_list":
            crayhints.append(f"{key}{separator}#{val}#")
        else:
            if key == "striping_unit":
                crayhints.append(f"cb_buffer_size{separator}{val}")
            if key == "romio_filesystem_type":
                os.environ.update({"ROMIO_FSTYPE_FORCE": val})
            # if key == "striping_factor":
            #     cb_multi = 1
            #     if "cray_cb_nodes_multiplier" in config_dict.keys():
            #         cb_multi = config_dict["cray_cb_nodes_multiplier"]
            #     crayhints.append(f"cb_nodes{separator}{val*cb_multi}")
            else:
                crayhints.append(f"{key}{separator}{val}")
    
    return ':'.join(map(str, crayhints))

    
def are_cray_hints_valid(config_dict, num_ranks, num_nodes):
    
    keys = config_dict.keys()
    # The combination of these configurations isn't valid
    if "cray_cb_write_lock_mode" in keys and config_dict["cray_cb_write_lock_mode"] == 1:
        if "romio_no_indep_rw" in keys and config_dict["romio_no_indep_rw"] == "false":
            return False
        if ("romio_cb_write" in keys and config_dict["romio_cb_write"] == "disable") or ("romio_cb_read" in keys and config_dict["romio_cb_read"] == "disable"):
            return False
    
    cb_multi = 1
    if "cray_cb_nodes_multiplier" in keys:
        cb_multi = config_dict["cray_cb_nodes_multiplier"]
    
    if "striping_factor" in keys:
        cb_nodes = config_dict["striping_factor"] * cb_multi
        if cb_nodes > num_ranks:
            return False
    
    # if "cb_config_list" in keys and "striping_factor" in keys:
        
    #     cb_ppn = int(config_dict["cb_config_list"].split(":")[1]) if config_dict["cb_config_list"].split(":")[1] != "*" else num_ranks//num_nodes
        
    #     cb_nodes = config_dict["striping_factor"] * cb_multi
        
    #     if min(cb_nodes, cb_ppn*num_nodes) > num_ranks:
    #         return False
    
    return True


