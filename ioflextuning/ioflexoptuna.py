# !pip install optuna
# Best Samplers for Optuna  Categorial RandSampler, GridSampler, TPESampler, NSGAIISampler, GPSampler, BruteForceSampler, Simulated Annealing
# GP requires torch  
import optuna
import logging
import subprocess
import shutil
import os
import sys
import time
import glob
import datetime
import signal
from shutil import move, rmtree
import re
import argparse
import shlex
import numpy
from datetime import datetime
import pickle

# Set of IO Configurations
## Lustre Striping
striping_factor = [4, 8, 16, 24, 32, 40, 48, 64, -1]
striping_unit = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728]

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
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def eval_func(trial):

    dir_path = os.environ['PWD']

    # Set IO Hints using IOFlex or ROMIO HINTS
    if ioflexset:
        config_file = open(os.path.join(dir_path,"config.conf"),'w')
    else:
        romio_path = os.path.join(dir_path,"romio-hints")
        config_file = open(romio_path,'w')

    # Set IO Configurations
    configs_str=""
    if len(striping_factor):
        strf_val = trial.suggest_categorical('striping_factor', striping_factor)
        if ioflexset:
            config_file.write("striping_factor = " + str(strf_val)+"\n")
        else:
            config_file.write("striping_factor " + str(strf_val)+"\n")
        configs_str += "striping_factor," + str(strf_val) + ","
    if len(striping_unit):
        stru_val = trial.suggest_categorical('striping_unit', striping_unit)
        if ioflexset:
            config_file.write("striping_unit = " + str(stru_val)+"\n")
            config_file.write("cb_buffer_size = " + str(stru_val)+"\n")
        else:
            config_file.write("striping_unit " + str(stru_val)+"\n")
            config_file.write("cb_buffer_size " + str(stru_val)+"\n")
        configs_str += "striping_unit," + str(stru_val) + ",cb_buffer_size," + str(stru_val) + ","  
    if len(cb_nodes):
        cbn_val = trial.suggest_categorical('cb_nodes', cb_nodes)
        if ioflexset:
            config_file.write("cb_nodes = " + str(cbn_val)+"\n")
        else:
            config_file.write("cb_nodes " + str(cbn_val)+"\n")
        configs_str += "cb_nodes," + str(cbn_val) + ","
    if len(romio_fs_type):
        fstype_val = trial.suggest_categorical('romio_fs_type', romio_fs_type)
        os.environ["ROMIO_FSTYPE_FORCE"]=fstype_val
        if ioflexset:
            config_file.write("romio_filesystem_type = " + fstype_val+"\n")
        else:
            config_file.write("romio_filesystem_type " + fstype_val+"\n")
        configs_str += "romio_filesystem_type," + fstype_val + ","
    if len(romio_ds_read):
        dsread_val = trial.suggest_categorical('romio_ds_read', romio_ds_read)
        if ioflexset:
            config_file.write("romio_ds_read = " + dsread_val+"\n")
        else:
            config_file.write("romio_ds_read " + dsread_val+"\n")
        configs_str += "romio_ds_read," + dsread_val + ","
    if len(romio_ds_write):
        dswrite_val = trial.suggest_categorical('romio_ds_write', romio_ds_write)
        if ioflexset:
            config_file.write("romio_ds_write = " + dswrite_val+"\n")
        else:
            config_file.write("romio_ds_write " + dswrite_val+"\n")
        configs_str += "romio_ds_write," + dswrite_val + ","
    if len(romio_cb_read):
        cbread_val = trial.suggest_categorical('romio_cb_read', romio_cb_read)
        if ioflexset:
            config_file.write("romio_cb_read = " + cbread_val+"\n")
        else:
            config_file.write("romio_cb_read " + cbread_val+"\n")
        configs_str += "romio_cb_read," + cbread_val + ","
    if len(romio_cb_write):
        cbwrite_val = trial.suggest_categorical('romio_cb_write', romio_cb_write)
        if ioflexset:
            config_file.write("romio_cb_write = " + cbwrite_val+"\n")
        else:
            config_file.write("romio_cb_write " + cbwrite_val +"\n")
        configs_str += "romio_cb_write," + cbwrite_val + ","
    if len(cb_config_list):
        cbl_val = trial.suggest_categorical('cb_config_list', cb_config_list)
        if ioflexset:
            config_file.write("cb_config_list = " + cbl_val+"\n")
        else:
            config_file.write("cb_config_list " + cbl_val+"\n")
        configs_str += "cb_config_list," + cbl_val + ","
    config_file.close()
    # Use ROMIO HINTS if IOFLex is not enabled
    if not ioflexset:
        os.environ['ROMIO_HINTS'] = romio_path

    # start application
    starttime = 0.0
    elapsedtime = 0.0

    env = os.environ.copy()
    starttime = time.time()
    q = subprocess.Popen(shlex.split(run_app), stdout=subprocess.PIPE, shell=False, cwd=os.getcwd())
    out, err = q.communicate()

    elapsedtime = time.time() - starttime
    # Write output of this config
    outline = configs_str + "elapsedtime," + str(elapsedtime)+"\n"
    outfile.write(outline)
    
    if logisset:
        logfile_o.write("Config: " + configs_str + "\n")
        out = str(out).replace(r'\n','\n')
        logfile_o.write(f'\n{out}')
        logfile_e.write("Config: " + configs_str + "\n")
        err = str(err).replace(r'\n','\n')
        logfile_e.write(f'\n{err}')

    # Clean application files after every iteration
    for f in files_to_clean:
        if os.path.isfile(f) or os.path.isdir(f):
            os.remove(f)
        if os.path.isdir(f):
            rmtree(f)
    print("Running config: " + outline)
    return elapsedtime


configs_space = {}

def ioflexoptuna():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--ioflex', action="store_true", default=False,help="IOFlex is enabled")

    ## TODO
    ap.add_argument('--outfile', type=str, help="Path to output CSV file", default="./OutIOFlexOptuna.csv")
    ap.add_argument('--inoptuna', type=str, help="Optuna study pickled file from previous runs", default=None)
    ap.add_argument('--outoptuna', '-o', type=str, help="Optuna study output pkl file", default="./optuna_study.pkl")
    ap.add_argument('--cmd', '-c', type=str, required=True, action="store", nargs="*", help="Application command line")
    # Optuna Options
    ap.add_argument("--max_trials", type=int, help="Max number of trials", default=50)
    ap.add_argument("--sampler", type=str, help="Sampler used by Optuna supported samplers: tpe (Tree-structured Parzen Estimator), rand (Random Search), gp (BO using Gaussian Processes), nsga (NSGAII), brute (Brute Force (all possibilities), grid (Grid))", default="tpe")
    ap.add_argument("--pruner", type=str, help="Pruner used by optuna", default=None)
    ap.add_argument("--with_log_path", type=str, help="If set the output of the program will be saved in this path (Default:None)", default=None)
    parse = ap.parse_args()
    args = vars(parse)
    global ioflexset
    ioflexset = args["ioflex"]

    global run_app
    run_app = " ".join(args["cmd"])

    ## TODO 
    global outfile
    outfile = open(args["outfile"], 'w')

    print("Starting Optuna Tuning")

    global max_trials, iter_count


    global applogpath, logisset, logfile_o, logfile_e

    applogpath = args["with_log_path"]
    if applogpath is None:
        logisset = False
    elif os.path.exists(applogpath):
        logisset = True
        time = datetime.now()
        logfile_o = open(os.path.join(applogpath, ("out." + time.strftime("%Y%m%d_%H.%M.%S"))), 'w')             
        logfile_e = open(os.path.join(applogpath, ("err." + time.strftime("%Y%m%d_%H.%M.%S"))), 'w')
    else:
        logisset = False
        print("The output of the app will not be logged")

    max_trials = args['max_trials']

    # Define search space for grid  sampler
    config_space = {}

    if len(striping_factor):
        config_space['striping_factor'] = striping_factor
    if len(striping_unit):
        config_space['striping_unit'] = striping_unit
    if len(cb_nodes):
        config_space['cb_nodes'] = cb_nodes
    if len(romio_fs_type):
        config_space['romio_fs_type'] = romio_fs_type
    if len(romio_ds_read):
        config_space['romio_ds_read'] = romio_ds_read

    if len(romio_ds_write):
        config_space['romio_ds_write'] = romio_ds_write

    if len(romio_cb_read):
        config_space['romio_cb_read'] = romio_cb_read

    if len(romio_cb_write):
        config_space['romio_cb_write'] = romio_cb_write

    if len(cb_config_list):
        config_space['cb_config_list'] = cb_config_list


    print(config_space)
    sampler = optuna.samplers.TPESampler()
    match args["sampler"]:
        case "gp":
            print("Running with Gaussian Processes")
            sampler = optuna.samplers.GPSampler()
        case "rand":
            print("Running with Random Search")
            sampler = optuna.samplers.RandomSampler()
        case "nsga":
            print("Running with NSGA-II")
            sampler = optuna.samplers.NSGAIISampler()
        case "grid":
            print("Running with Grid Search")
            sampler = optuna.samplers.GridSampler(config_space)
        case "brute":
            print("Running with Brute Force")
            sampler = optuna.samplers.BruteForceSampler()

    # Load previous optuna instance
    if args["inoptuna"] is None:
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name='optuna_study')
    else:
        with open(args["inoptuna"], 'rb') as f:
            study = pickle.load(f)

    
    # Perform Optimization
    study.optimize(eval_func, n_trials=args["max_trials"])

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Time: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("Saving Results")


    with open(args["outoptuna"], 'wb') as f:
        pickle.dump(study, f)

    outfile.close()

    if logisset:
        logfile_o.close()
        logfile_e.close()

if __name__ == '__main__':
    ioflexoptuna()