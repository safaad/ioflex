import os
import glob
import shutil
import optuna
import nevergrad as ng
import numpy as np
import darshan
import json
from darshan.backend.cffi_backend import accumulate_records

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
    "nop": optuna.pruners.NopPruner,
}

# NNEVERGRAD SPECIFIC
OPTIMIZER_MAP = {
    "ngioh": ("NgIohTuned", ng.optimizers.NgIoh),
    "twopde": ("Two Points Differential Evolution", ng.optimizers.TwoPointsDE),
    "pdopo": ("PortfolioDiscreteOnePlusOne", ng.optimizers.PortfolioDiscreteOnePlusOne),
    "tbpsa": (
        "TBPSA Test-based population-size adaptation",
        ng.optimizers.registry["TBPSA"],
    ),
    "ngopt": ("NGOpt", ng.optimizers.NGOpt),
}

# IO CONFIGURATIONS Parameters Space
CONFIG_ROMIO_KEY = {
    "romio_filesystem_type",
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
    "striping_factor",
    "striping_unit",
    "cb_nodes",
    "cb_config_list",
    "romio_no_indep_rw",
}

CONFIG_CRAY_KEY = {
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
    "striping_factor",
    "striping_unit",
    "cb_nodes",
    "cray_cb_nodes_multiplier",
    "cray_cb_write_lock_mode",
    "direct_io",
    "aggregator_placement_stride",
    "romio_no_indep_rw",
}

CONFIG_OMPIO_KEY = {
    # TODO
    "striping_unit"
}


def get_config_map(hints, config_path):
    CONFIG_KEY = {
        "cray": CONFIG_CRAY_KEY,
        "ompio": CONFIG_OMPIO_KEY,
        "romio": CONFIG_ROMIO_KEY,
    }.get(
        hints, CONFIG_ROMIO_KEY
    )  # Use ROMIO as default fallback
    

    
    try:
        with open(config_path, 'r') as file:
            configdata = json.load(file)
    except FileNotFoundError:
        print("Error: The file config.json was not found.")
    
    files_to_clean = configdata["files_to_clean"]
    files_to_stripe = configdata["files_to_stripe"]
    
    CONFIG_MAP = {}
    for key, values in configdata.items():
        if key == "files_to_clean" or key == "files_to_stripe":
            continue
        if key not in CONFIG_KEY:
            raise KeyError("Invalid or unsupported hint")
        CONFIG_MAP[key] = values
    
    return CONFIG_MAP, files_to_clean, files_to_stripe


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


def set_hints_env_cray(config_dict):

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
            if key == "striping_factor":
                # cb_multi = config_dict["cray_cb_nodes_multiplier"] if "cray_cb_nodes_multiplier" in config_dict.keys() else 1
                crayhints.append(f"striping_factor{separator}{val}")
            else:
                crayhints.append(f"{key}{separator}{val}")

    return ":".join(map(str, crayhints))


def are_cray_hints_valid(config_dict, num_ranks, num_nodes):

    keys = config_dict.keys()
    # The combination of these configurations isn't valid
    if (
        "cray_cb_write_lock_mode" in keys
        and config_dict["cray_cb_write_lock_mode"] == 1
    ):
        if "romio_no_indep_rw" in keys and config_dict["romio_no_indep_rw"] == "false":
            return False
        if (
            "romio_cb_write" in keys and config_dict["romio_cb_write"] == "disable"
        ) or ("romio_cb_read" in keys and config_dict["romio_cb_read"] == "disable"):
            return False

    cb_multi = (
        config_dict["cray_cb_nodes_multiplier"]
        if "cray_cb_nodes_multiplier" in keys
        else 1
    )
    cb_nodes = (
        config_dict["striping_factor"] * cb_multi
        if "striping_factor" in keys
        else config_dict["cb_nodes"]
    )
    if cb_nodes > num_ranks:
        return False

    # if "cb_config_list" in keys and "striping_factor" in keys:

    #     cb_ppn = int(config_dict["cb_config_list"].split(":")[1]) if config_dict["cb_config_list"].split(":")[1] != "*" else num_ranks//num_nodes

    #     cb_nodes = config_dict["striping_factor"] * cb_multi

    #     if min(cb_nodes, cb_ppn*num_nodes) > num_ranks:
    #         return False

    return True


def get_bandwidth_darshan(log_path, mod):

    darshan_files = glob.glob(log_path)
    if len(darshan_files) == 0:
        print("No darshan file generated")
        return -1
    log_file = darshan_files[0]
    report = darshan.DarshanReport(log_file, read_all=False)
    if mod not in report.modules or report.modules[mod]["len"] == 0:
        os.remove(log_file)
        print("Empty Darshan File")
        return -1
    report.mod_read_all_records(mod)
    recs = report.records[mod].to_df()
    acc_rec = accumulate_records(recs, mod, report.metadata["job"]["nprocs"])

    bandwidth_slowest = acc_rec.derived_metrics.agg_perf_by_slowest
    os.remove(log_file)

    return bandwidth_slowest


def remove_path(path_pattern: str):

    matches = glob.glob(path_pattern, recursive=True)

    if not matches:
        print(f"No matches found for: {path_pattern}")
        return
    for path in matches:
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
                print(f"Removed file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                print(f"Skipping unkown type: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
