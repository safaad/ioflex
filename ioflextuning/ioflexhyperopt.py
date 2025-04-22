# pip install hyperopt, pyll for gp
import os
import sys
import time
import argparse
import shlex
import subprocess
import logging
import joblib
import functools
from datetime import datetime
from shutil import rmtree
from hyperopt import fmin, tpe, Trials, rand, hp, anneal
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ioflexpredict import ioflexpredict
from ioflexheader import CONFIG_MAP
from ioflexsetstriping import setstriping
from utils import header 

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Application Specific
files_to_stripe = []
files_to_clean = []

def eval_func(params, model):

    dir_path = os.getcwd()
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    with open(config_path, "w") as config_file:
        sample_instance = {}
        configs_list = []
        
        for key, values in CONFIG_MAP.items():
            if not values:
                continue
            
            selected_val = params[key]
            separator = " = " if ioflexset else " "
            sample_instance[key] = selected_val
            print(sample_instance)
            
            configs_list.append(str(selected_val))
            # Special handling for specific keys
            match key:
                case "striping_factor":
                    stripe_count = int(selected_val) 
                case "striping_unit":
                        stripe_size = (str(selected_val // 1048576) + "M")
                        config_file.write(f"cb_buffer_size{separator}{selected_val}\n")
                case "romio_filesystem_type":
                    os.environ["ROMIO_FSTYPE_FORCE"] = selected_val
                case "cb_config_list":
                    if ioflexset:
                        config_file.write(f"{key}{separator}\"{selected_val}\"\n")
                    continue
            config_file.write(f"{key}{separator}{selected_val}\n")
            
    configs_str = ",".join(configs_list)
    if not ioflexset:
        os.environ["ROMIO_HINTS"] = config_path
    else:
        os.environ["IOFLEX_HINTS"] = config_path

    start_time = time.time()
    process = subprocess.Popen(shlex.split(run_app), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    elapsedtime = time.time() - start_time
    # set striping with lfs setstripe
    for f in files_to_stripe:
        setstriping(f, stripe_count, stripe_size)

    if model:
        predtime = ioflexpredict.predict_instance(model, sample_instance)
        outline = f"{configs_str},{elapsedtime},{predtime}\n"
    else:
        outline = f"{configs_str},{elapsedtime}\n"
    
    outfile.write(outline)
    
    if logisset:
        logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
        logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")
    
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f) if os.path.isfile(f) else rmtree(f)
    
    logger.info(f"Running config: {outline}")
    
    return elapsedtime


configs_space = {}


def ioflexhyperopt():

    ap = argparse.ArgumentParser(description="Hyperparameter Optimization using Hyperopt")
    ap.add_argument("--ioflex", action="store_true", default=False, help="Enable IOFlex")
    ap.add_argument("--outfile", type=str, default="./OutIOFlexHyperopt.csv", help="Path to output CSV file")
    ap.add_argument("--inhyperopt", type=str, default=None, help="Pickled Hyperopt instance from previous runs")
    ap.add_argument("--outhyperopt", "-o", type=str, default="./hyperopt_trails.pkl", help="Output pickle file for Hyperopt instance")
    ap.add_argument("--cmd", "-c", type=str, required=True, nargs="*", help="Application command line")
    ap.add_argument("--max_evals", type=int, default=50, help="Max number of evaluations")
    ap.add_argument("--algorithm", type=str, choices=["tpe", "rand", "anneal"], default="tpe",
                    help="Hyperopt Algorithm: tpe (Tree-structured Parzen Estimator), rand (Random Search), anneal (Simulated Annealing)")
    ap.add_argument("--with_log_path", type=str, default=None, help="Path for logging program output")
    ap.add_argument("--with_model", type=str, default=None, help="Path to trained prediction model")
    args = vars(ap.parse_args())
    
    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], "w")

    # global model
    model = joblib.load(args["with_model"]) if args["with_model"] else None
    
    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    print("Starting HyperOpt Tuning")

    configs_space = {key: hp.choice(key, values) for key, values in CONFIG_MAP.items() if values}


    header_items = []   
    for key, values in CONFIG_MAP.items():
        if values:
            header_items.append(key)

    header_items.append("elapsedtime")
    if args["with_model"]:
        header_items.append("predicttime")

    header_str = ",".join(header_items) + "\n"
    outfile.write(header_str)
    
    algo_dict = {
        "tpe": tpe.suggest,
        "rand": rand.suggest,
        "anneal": anneal.suggest,
    }
    
    algo = algo_dict.get(args["algorithm"], tpe.suggest)

    trials = Trials() if args["inhyperopt"] is None else joblib.load(args["inhyperopt"])
    eval_func_partial = functools.partial(eval_func, model=model)
    best = fmin(fn=eval_func_partial, space=configs_space, algo=algo, max_evals=args["max_evals"], trials=trials)
    
    logger.info(f"Best parameters: {best}")
    joblib.dump(trials, args["outhyperopt"])
    outfile.close()
    
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    header.printheader()
    ioflexhyperopt()
