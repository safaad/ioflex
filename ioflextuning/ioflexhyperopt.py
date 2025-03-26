# pip install hyperopt, pyll for gp
import os
import sys
import time
import argparse
import shlex
import subprocess
import logging
import joblib
from datetime import datetime
from shutil import rmtree
from hyperopt import fmin, tpe, Trials, rand, hp, anneal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set of IO Configurations
CONFIG_ENTRIES = {
    "striping_factor": [4, 8, 16, 24, 32, 40, 48, 64, -1],
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
    "cb_nodes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "cb_config_list": [],
    "romio_fs_type": ["LUSTRE:", "UFS:"],
    "romio_ds_read": ["automatic", "enable", "disable"],
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"]
}

files_to_clean = []

def eval_func(params):
    dir_path = os.getcwd()
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    with open(config_path, "w") as config_file:
        configs_str = ""
        for key, values in CONFIG_ENTRIES.items():
            if values:
                selected_val = params[key]
                if ioflexset:
                    config_file.write(f"{key} = {selected_val}\n")
                else:
                    config_file.write(f"{key} {selected_val}\n")
                configs_str += f"{selected_val},"
                if key == "striping_unit":
                    config_file.write(f"cb_buffer_size = {selected_val}\n")
                if key == "romio_fs_type":
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


configs_space = {}


def ioflexhyperopt():

    ap = argparse.ArgumentParser(description="Hyperparameter Optimization using Hyperopt")
    ap.add_argument("--ioflex", action="store_true", default=False, help="Enable IOFlex")
    ap.add_argument("--outfile", type=str, default="./OutIOFlexHyperopt.csv", help="Path to output CSV file")
    ap.add_argument("--inhyperopt", type=str, default=None, help="Pickled Hyperopt instance from previous runs")
    ap.add_argument("--outhyperopt", "-o", type=str, default="./hyperopt_trails.pkl", help="Output pickle file for Hyperopt instance")
    ap.add_argument("--cmd", "-c", type=str, required=True, nargs="*", help="Application command line")
    ap.add_argument("--max_evals", type=int, default=50, help="Max number of evaluations")
    ap.add_argument("--algorithm", type=str, choices=["tpe", "rand", "anneal", "gp"], default="tpe",
                    help="Hyperopt Algorithm: tpe (Tree-structured Parzen Estimator), rand (Random Search), anneal (Simulated Annealing), gp (Gaussian Process)")
    ap.add_argument("--with_log_path", type=str, default=None, help="Path for logging program output")
    
    args = vars(ap.parse_args())
    
    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], "w")

    global run_app
    run_app = " ".join(args["cmd"])

    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    print("Starting HyperOpt Tuning")

    global max_evals, iter_count

    global no_of_configs
    no_of_configs = 0

    configs_space = {key: hp.choice(key, values) for key, values in CONFIG_ENTRIES.items() if values}

    algo_dict = {
        "tpe": tpe.suggest,
        "rand": rand.suggest,
        "anneal": anneal.suggest
    }
    
    algo = algo_dict.get(args["algorithm"], tpe.suggest)

    trials = Trials() if args["inhyperopt"] is None else joblib.load(args["inhyperopt"])

    best = fmin(fn=eval_func, space=configs_space, algo=algo, max_evals=args["max_evals"], trials=trials)
    
    logger.info(f"Best parameters: {best}")
    joblib.dump(trials, args["outhyperopt"])
    outfile.close()
    
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    ioflexhyperopt()
