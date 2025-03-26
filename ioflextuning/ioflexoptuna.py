# !pip install optuna
# Best Samplers for Optuna  Categorial RandSampler, GridSampler, TPESampler, NSGAIISampler, GPSampler, BruteForceSampler, Simulated Annealing
# GP requires torch
import os
import sys
import time
import argparse
import shlex
import subprocess
import joblib
import optuna
import joblib
import logging
from datetime import datetime
from shutil import rmtree

# Set of IO Configurations
striping_factor = [4, 8, 16, 24, 32, 40, 48, 64, -1]
striping_unit = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728]
cb_nodes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
cb_config_list = []
romio_fs_type = ["LUSTRE:", "UFS:"]
romio_ds_read = ["automatic", "enable", "disable"]
romio_ds_write = ["automatic", "enable", "disable"]
romio_cb_read = ["automatic", "enable", "disable"]
romio_cb_write = ["automatic", "enable", "disable"]


# Application Specific
files_to_clean = []

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def eval_func(trial):

    dir_path = os.getcwd()
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

    start_time = time.time()
    process = subprocess.Popen(shlex.split(run_app), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    elapsed_time = time.time() - start_time
    
    outline = f"{configs_str}{elapsed_time}\n"
    outfile.write(outline)
    
    if logisset:
        logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
        logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")
    
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f) if os.path.isfile(f) else rmtree(f)
    
    logger.info(f"Running config: {outline}")
    
    return elapsed_time



def ioflexoptuna():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ioflex", action="store_true", default=False, help="Enable IOFlex")
    ap.add_argument("--outfile", type=str, default="./OutIOFlexOptuna.csv", help="Path to output CSV file")
    ap.add_argument("--inoptuna", type=str, default=None, help="Optuna study pickled file")
    ap.add_argument("--outoptuna", "-o", type=str, default="./optuna_study.pkl", help="Optuna study output pkl file")
    ap.add_argument("--cmd", "-c", type=str, required=True, nargs="*", help="Application command line")
    ap.add_argument("--max_trials", type=int, default=50, help="Max number of trials")
    ap.add_argument("--sampler", type=str, choices=["tpe", "rand", "gp", "nsga", "brute", "grid", "auto"], default="tpe", help="Optuna sampler")
    ap.add_argument("--pruner", type=str, choices=["hyper", "median", "successivehalving", "nop"], default="nop", help="Optuna pruner")
    ap.add_argument("--with_log_path", type=str, default=None, help="Path for logging output")
    args = vars(ap.parse_args())
    
    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], "w")
    
    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    sampler = getattr(optuna.samplers, f"{args['sampler'].capitalize()}Sampler")()
    pruner = getattr(optuna.pruners, f"{args['pruner'].capitalize()}Pruner", optuna.pruners.NopPruner)()
    
    if args["inoptuna"]:
        study = joblib.load(args["inoptuna"])
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    
    study.optimize(eval_func, n_trials=args["max_trials"])
    
    logger.info(f"Best trial: {study.best_trial}")
    joblib.dump(study, args["outoptuna"])

    outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    ioflexoptuna()
