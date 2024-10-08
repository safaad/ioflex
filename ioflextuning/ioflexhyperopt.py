# pip install hyperopt, pyll for gp
from hyperopt import fmin, tpe, Trials, rand, hp, anneal, rand
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


def eval_func(params):

    dir_path = os.environ["PWD"]

    # Set IO Hints using IOFlex or ROMIO HINTS
    if ioflexset:
        config_file = open(os.path.join(dir_path, "config.conf"), "w")
    else:
        romio_path = os.path.join(dir_path, "romio-hints")
        config_file = open(romio_path, "w")

    # Set IO Configurations
    configs_str = ""
    if strf_id is not None:
        strf_val = params["striping_factor"]
        if ioflexset:
            config_file.write("striping_factor = " + str(strf_val) + "\n")
        else:
            config_file.write("striping_factor " + str(strf_val) + "\n")
        configs_str += str(strf_val) + ","
    if stru_id is not None:
        stru_val = params["striping_unit"]
        if ioflexset:
            config_file.write("striping_unit = " + str(stru_val) + "\n")
            config_file.write("cb_buffer_size = " + str(stru_val) + "\n")
        else:
            config_file.write("striping_unit " + str(stru_val) + "\n")
            config_file.write("cb_buffer_size " + str(stru_val) + "\n")
        configs_str += str(stru_val) + "," + str(stru_val) + ","
    if cbn_id is not None:
        cbn_val = params["cb_nodes"]
        if ioflexset:
            config_file.write("cb_nodes = " + str(cbn_val) + "\n")
        else:
            config_file.write("cb_nodes " + str(cbn_val) + "\n")
        configs_str += str(cbn_val) + ","
    if fstype_id is not None:
        fstype_val = params["romio_fs_type"]
        os.environ["ROMIO_FSTYPE_FORCE"] = fstype_val
        if ioflexset:
            config_file.write("romio_filesystem_type = " + fstype_val + "\n")
        else:
            config_file.write("romio_filesystem_type " + fstype_val + "\n")
        configs_str += fstype_val + ","
    if dsread_id is not None:
        dsread_val = params["romio_ds_read"]
        if ioflexset:
            config_file.write("romio_ds_read = " + dsread_val + "\n")
        else:
            config_file.write("romio_ds_read " + dsread_val + "\n")
        configs_str += dsread_val + ","
    if dswrite_id is not None:
        dswrite_val = params["romio_ds_write"]
        if ioflexset:
            config_file.write("romio_ds_write = " + dswrite_val + "\n")
        else:
            config_file.write("romio_ds_write " + dswrite_val + "\n")
        configs_str += dswrite_val + ","
    if cbread_id is not None:
        cbread_val = params["romio_cb_read"]
        if ioflexset:
            config_file.write("romio_cb_read = " + cbread_val + "\n")
        else:
            config_file.write("romio_cb_read " + cbread_val + "\n")
        configs_str += cbread_val + ","
    if cbwrite_id is not None:
        cbwrite_val = params["romio_cb_write"]
        if ioflexset:
            config_file.write("romio_cb_write = " + cbwrite_val + "\n")
        else:
            config_file.write("romio_cb_write " + cbwrite_val + "\n")
        configs_str += cbwrite_val + ","
    if cbl_id is not None:
        cbl_val = params["cb_config_list"]
        if ioflexset:
            config_file.write("cb_config_list = " + cbl_val + "\n")
        else:
            config_file.write("cb_config_list " + cbl_val + "\n")
        configs_str += cbl_val + ","
    config_file.close()
    # Use ROMIO HINTS if IOFLex is not enabled
    if not ioflexset:
        os.environ["ROMIO_HINTS"] = romio_path

    # start application
    starttime = 0.0
    elapsedtime = 0.0

    env = os.environ.copy()
    starttime = time.time()
    q = subprocess.Popen(
        shlex.split(run_app), stdout=subprocess.PIPE, shell=False, cwd=os.getcwd()
    )
    out, err = q.communicate()

    elapsedtime = time.time() - starttime
    # Write output of this config

    outline = configs_str + str(elapsedtime) + "\n"
    outfile.write(outline)

    if logisset:
        logfile_o.write("Config: " + configs_str + "\n")
        out = str(out).replace(r"\n", "\n")
        logfile_o.write(f"\n{out}")
        logfile_e.write("Config: " + configs_str + "\n")
        err = str(err).replace(r"\n", "\n")
        logfile_e.write(f"\n{err}")

    # Clean application files after every iteration
    for f in files_to_clean:
        if os.path.isfile(f) or os.path.isdir(f):
            os.remove(f)
        if os.path.isdir(f):
            rmtree(f)
    print("Running config: " + outline)
    return elapsedtime


configs_space = {}


def ioflexhyperopt():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="IOFlex is enabled"
    )

    ## TODO
    ap.add_argument(
        "--outfile",
        type=str,
        help="Path to output CSV file",
        default="./OutIOFlexHyperopt.csv",
    )
    ap.add_argument(
        "--inhyperopt",
        type=str,
        help="Hyperopt instance pickled file from previous runs",
        default=None,
    )
    ap.add_argument(
        "--outhyperopt",
        "-o",
        type=str,
        help="Hyperopt instance output pkl file",
        default="./hyperopt_trails.pkl",
    )
    ap.add_argument(
        "--cmd",
        "-c",
        type=str,
        required=True,
        action="store",
        nargs="*",
        help="Application command line",
    )
    # Hyperopt Options
    ap.add_argument(
        "--max_evals", type=int, help="Max number of evaluations", default=50
    )
    ap.add_argument(
        "--algorithm",
        type=str,
        choices=["tpe", "rand", "anneal", "gp"],
        help="Algorithm used by Hyperopt possible options: tpe (Tree-structured Parzen Estimator), rand (Random Search), anneal (Simulated Annealing), gp (BO using Gaussian Processes)",
        default="tpe",
    )
    ap.add_argument(
        "--with_log_path",
        type=str,
        help="If set the output of the program will be saved in this path (Default:None)",
        default=None,
    )
    parse = ap.parse_args()
    args = vars(parse)
    global ioflexset
    ioflexset = args["ioflex"]

    global run_app
    run_app = " ".join(args["cmd"])

    ## TODO
    global outfile
    outfile = open(args["outfile"], "w")

    print("Starting HyperOpt Tuning")

    global max_evals, iter_count

    global no_of_configs
    no_of_configs = 0

    global strf_id, stru_id, cbn_id, fstype_id, dsread_id, dswrite_id, cbread_id, cbwrite_id, cbl_id
    strf_id = None
    stru_id = None
    cbn_id = None
    fstype_id = None
    dsread_id = None
    dswrite_id = None
    cbread_id = None
    cbwrite_id = None
    cbl_id = None
    header_str = ""
    if len(striping_factor):
        configs_space.update(
            {"striping_factor": hp.choice("striping_factor", striping_factor)}
        )
        strf_id = no_of_configs
        no_of_configs += 1
        header_str += "striping_factor,"
    if len(striping_unit):
        configs_space.update(
            {"striping_unit": hp.choice("striping_unit", striping_unit)}
        )
        stru_id = no_of_configs
        no_of_configs += 1
        header_str += "striping_unit,cb_buffer_size,"
    if len(cb_nodes):
        configs_space.update({"cb_nodes": hp.choice("cb_nodes", cb_nodes)})
        cbn_id = no_of_configs
        no_of_configs += 1
        header_str += "cb_nodes,"
    if len(romio_fs_type):
        configs_space.update(
            {"romio_fs_type": hp.choice("romio_fs_type", romio_fs_type)}
        )
        fstype_id = no_of_configs
        no_of_configs += 1
        header_str += "romio_filesystem_type,"
    if len(romio_ds_read):
        configs_space.update(
            {"romio_ds_read": hp.choice("romio_ds_read", romio_ds_read)}
        )
        dsread_id = no_of_configs
        no_of_configs += 1
        header_str += "romio_ds_read,"
    if len(romio_ds_write):
        configs_space.update(
            {"romio_ds_write": hp.choice("romio_ds_write", romio_ds_write)}
        )
        dswrite_id = no_of_configs
        no_of_configs += 1
        header_str += "romio_ds_write,"
    if len(romio_cb_read):
        configs_space.update(
            {"romio_cb_read": hp.choice("romio_cb_read", romio_cb_read)}
        )
        cbread_id = no_of_configs
        no_of_configs += 1
        header_str += "romio_cb_read,"
    if len(romio_cb_write):
        configs_space.update(
            {"romio_cb_write": hp.choice("romio_cb_write", romio_cb_write)}
        )
        cbwrite_id = no_of_configs
        no_of_configs += 1
        header_str += "romio_cb_write,"
    if len(cb_config_list):
        configs_space.update(
            {"cb_config_list": hp.choice("cb_config_list", cb_config_list)}
        )
        cbl_id = no_of_configs
        no_of_configs += 1
        header_str += "cb_config_list,"

    header_str += "elapsedtime\n"
    outfile.write(header_str)
    global applogpath, logisset, logfile_o, logfile_e

    applogpath = args["with_log_path"]
    if applogpath is None:
        logisset = False
    elif os.path.exists(applogpath):
        logisset = True
        time = datetime.now()
        logfile_o = open(
            os.path.join(applogpath, ("out." + time.strftime("%Y%m%d_%H.%M.%S"))), "w"
        )
        logfile_e = open(
            os.path.join(applogpath, ("err." + time.strftime("%Y%m%d_%H.%M.%S"))), "w"
        )
    else:
        logisset = False
        print("The output of the app will not be logged")

    max_evals = args["max_evals"]

    algo = tpe.suggest

    match args["algorithm"]:
        case "anneal":
            print("Running with Simulated Annealing ")
            algo = anneal.suggest
        case "rand":
            print("Running with Random Search")
            algo = rand.suggest
        case "gp":
            print("Running with Bayesian Optimization using Gaussian Processes")
            algo = gp.suggest
        case "tpe":
            print("Running with TPE")
            algo = tpe.suggest

    # Load previous hyperopt instance
    if args["inhyperopt"] is None:
        trials = Trials()
    else:
        with open(args["inhyperopt"], "rb") as infile:
            trials = pickle.load(infile)

    # Perform Optimization

    best = fmin(
        fn=eval_func, space=configs_space, algo=algo, max_evals=max_evals, trials=trials
    )

    print("Best parameters: ", best)

    print("Saving Results")

    with open(args["outhyperopt"], "wb") as f:
        pickle.dump(trials, f)

    outfile.close()

    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    ioflexhyperopt()
