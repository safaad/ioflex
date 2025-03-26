import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
import argparse
import optuna
import logging
import subprocess
import os
import sys
import time
from shutil import rmtree
import argparse
import shlex
from datetime import datetime
import joblib



# Set of IO Configurations
## Lustre Striping
striping_factor = [4, 8, 16, 24, 32, 40, 48, 64, -1]
striping_unit = [
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
]

## Collective Buffering
### Adjust number of cb_nodes to total number of nodes
cb_nodes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
cb_config_list = []

## ROMIO Optimizations
romio_fs_type = ["LUSTRE:", "UFS:"]
romio_ds_read = ["automatic", "enable", "disable"]
romio_ds_write = ["automatic", "enable", "disable"]
romio_cb_read = ["automatic", "enable", "disable"]
romio_cb_write = ["automatic", "enable", "disable"]

iter_count = 0
# Application Specific
# List of files to clean after running application
files_to_clean = []


# Configure Optuna logging to stdout
optuna.logging.set_verbosity(optuna.logging.INFO)
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

config_cols = [
    "romio_filesystem_type",
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
    "striping_factor",
    "striping_unit",
    "cb_nodes",
    "cb_config_list"
]

def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean(numerator / denominator) * 100
    return smape_value

def predict_instance(model, category_mappings, instance):
    
    for col, mapping in category_mappings.items():
        instance[col] = mapping.get(instance[col], -1)

    instance_df = pd.DataFrame([instance])
    prediction = model.predict(instance_df)[0]
    return prediction

def eval_func(model, category_mappings, trial):
    
    dir_path = os.environ.get("PWD", os.getcwd())
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")
    with open(config_path, "w") as config_file:
        sample_instance = {}
        config_entries = {
            "striping_factor": striping_factor,
            "striping_unit": striping_unit,
            "cb_nodes": cb_nodes,
            "romio_filesystem_type": romio_fs_type,
            "romio_ds_read": romio_ds_read,
            "romio_ds_write": romio_ds_write,
            "romio_cb_read": romio_cb_read,
            "romio_cb_write": romio_cb_write,
            "cb_config_list": cb_config_list
        }

        configs_str = ""
        for key, values in config_entries.items():
            if values:
                selected_val = trial.suggest_categorical(key, values)
                if ioflexset:
                    config_file.write(f"{key} = {selected_val}\n")
                else:
                    config_file.write(f"{key} {selected_val}\n")
                    
                configs_str += str(selected_val) + ","
                sample_instance[key] = selected_val
                if key == "striping_unit":
                    config_file.write(f"cb_buffer_size = {selected_val}\n")
                if key == "romio_filesystem_type":
                    os.environ["ROMIO_FSTYPE_FORCE"] = selected_val
    
    # Use ROMIO HINTS if IOFLex is not enabled
    if not ioflexset:
        os.environ["ROMIO_HINTS"] = config_path

    # start application
    starttime = time.time()
    q = subprocess.Popen(
        shlex.split(run_app), stdout=subprocess.PIPE, shell=False, cwd=os.getcwd()
    )
    out, err = q.communicate()
    elapsedtime = time.time() - starttime
    
    predtime = predict_instance(model, category_mappings, sample_instance)
    outline = f"{configs_str}{elapsedtime},{predtime}\n"
    outfile.write(outline)

    if logisset:
        logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
        logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")

    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f) if os.path.isfile(f) else rmtree(f)
            
    print("Running config: " + outline)
    return elapsedtime


configs_space = {}

def ioflexpredict():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ioflex", action="store_true", default=False, help="IOFlex is enabled")
    ap.add_argument("--outfile", type=str, default="./OutIOFlexOptuna.csv", help="Path to output CSV file")
    ap.add_argument("--inoptuna", type=str, default=None, help="Optuna study pickled file from previous runs")
    ap.add_argument("--outoptuna", "-o", type=str, default="./optuna_study.pkl", help="Optuna study output pkl file")
    ap.add_argument("--cmd", "-c", type=str, required=True, nargs="*", help="Application command line")
    ap.add_argument("--max_trials", type=int, default=50, help="Max number of trials")
    ap.add_argument("--sampler", type=str, choices=["tpe", "rand", "gp", "nsga", "brute", "grid", "auto"], default="tpe", help="Sampler used by Optuna")
    ap.add_argument("--trained-model", type=str, required=True, help="Path to the trained model")
    ap.add_argument("--with_log_path", type=str, default=None, help="Path for logging application output")
    ap.add_argument("--cat-mapping", type=str, required=True, help="Input category mappings")
    args = vars(ap.parse_args())
    
    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], "w")
    
    category_mappings = joblib.load(args["cat-mapping"])
    model = joblib.load(args["trained-model"])
    
    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")
    
    print("Starting Optuna Tuning")
    
    config_space = {key: globals()[key] for key in [
        "striping_factor", "striping_unit", "cb_nodes", "romio_fs_type",
        "romio_ds_read", "romio_ds_write", "romio_cb_read", "romio_cb_write", "cb_config_list"
    ] if len(globals()[key]) > 0}
    
    header_str = ",".join(config_space.keys()) + ",pred-time,real-time\n"
    outfile.write(header_str)
    
    samplers = {
        "tpe": optuna.samplers.TPESampler(),
        "gp": optuna.samplers.GPSampler(),
        "rand": optuna.samplers.RandomSampler(),
        "nsga": optuna.samplers.NSGAIISampler(),
        "grid": optuna.samplers.GridSampler(config_space),
        "brute": optuna.samplers.BruteForceSampler(),
        "auto": optuna.create_study().sampler,
    }
    
    sampler = samplers.get(args["sampler"], optuna.samplers.TPESampler())
    
    if args["inoptuna"]:
        study = joblib.load(args["inoptuna"])
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name="optuna_study")
    
    study.optimize(lambda trial: eval_func(trial, model, category_mappings), n_trials=args["max_trials"])
    
    print("Best trial:", study.best_trial)
    joblib.dump(study, args["outoptuna"])
    outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    ioflexpredict()
