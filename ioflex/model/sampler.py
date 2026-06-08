import os
import sys
import time
import argparse
import shlex
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import math
from datetime import datetime
from shutil import rmtree
from scipy.stats import qmc
import ioflex.model.parser as parser
from tqdm.auto import tqdm
from ioflex.common import (
    get_config_map,
    set_hints_with_ioflex,
    set_hints_env_romio,
    set_hints_env_cray,
    repair_cray_hints_valid,
    get_bandwidth_darshan,
    remove_path,
)

from ioflex.striping import setstriping


def round_to_nearest_square(n):
    return round(math.sqrt(n)) ** 2

def generate_lhs_samples(config_space, nsamples):
    
    param_keys = config_space.keys()
    param_grid = config_space.values()
    n_dims = len(param_grid)
    lhs = qmc.LatinHypercube(d=n_dims, optimization='lloyd')

    print(f"Number of samples is: ",nsamples)
    lhs_samples = lhs.random(n=nsamples)

    sampled_configs = []
    for i in range(nsamples):
        sampled_config = []
        for j, param_values in enumerate(param_grid):
            index = int(np.floor(lhs_samples[i, j] * len(param_values)))
            sampled_config.append(param_values[index])
        sampled_configs.append(dict(zip(param_keys, sampled_config)))

    return sampled_configs


def eval_runs(samples_config, nsamples):

    global outfile, out_is_open
    dir_path = os.environ.get("PWD", os.getcwd())
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    for i in range(nsamples):
        sample_instance = samples_config[i]
        if hints == "cray":
            # if options are not valid skip the trial
            repair_cray_hints_valid(sample_instance, num_ranks, num_nodes)


        if ioflexset:
            set_hints_with_ioflex(sample_instance, config_path)
            os.environ["IOFLEX_HINTS"] = config_path
        else:
            if hints == "romio":
                set_hints_env_romio(sample_instance, config_path)
                os.environ["ROMIO_HINTS"] = config_path
            if hints == "cray":
                crayhints = set_hints_env_cray(sample_instance)
                os.environ["MPICH_MPIIO_HINTS"] = crayhints
            # if hints == "ompio":
            #  TODO


        stripe_count = (
            int(sample_instance["striping_factor"])
            if "striping_factor" in sample_instance
            else 8
        )
        stripe_size = (
            str(sample_instance["striping_unit"] // 1048576) + "M"
            if "striping_unit" in sample_instance
            else "1M"
        )
        # Set striping with lfs setstripe
        for f in files_to_stripe:
            setstriping(f, stripe_count, stripe_size)

        start_time = time.time()
        process = subprocess.Popen(
            shlex.split(run_app),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=os.getcwd(),
        )
        out, err = process.communicate()
        objective = time.time() - start_time

        sample_instance["elapsedtime"] = objective
        if tune_bandwidth:
            darshan_dir = os.environ["DARSHAN_LOG_DIR_PATH"]
            log_path = os.path.join(darshan_dir, "*.darshan")
            # get MPI-IO bandwidth MiB/s
            objective = get_bandwidth_darshan(log_path, "MPI-IO")
            if objective == -1:
                print(
                    "Darshan file wasn't properly generated. Check the Darshan settings or the application correctness"
                )
                continue
            sample_instance["I/O-Bandwidth-Mib/s"] = objective

        if not out_is_open:
            try:
                outfile = open(outfilepath, "w")
                out_is_open = True
                outfile.write(",".join(sample_instance.keys()) + "\n")
            except:
                raise Exception("Cannot create file ", outfilepath)
        outfile.write(",".join(map(str, sample_instance.values())) + "\n")

        configs_str = ", ".join(f"{k}: {v}" for k, v in sample_instance.items())
        if logisset:
            logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
            logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")
        print(f"Running config: {configs_str}")

        for f in files_to_clean:
            remove_path(f)


def run(args=None):

    ap = argparse.ArgumentParser(prog="ioflex model --sample")
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./OutLHSsampler.csv",
        help="Path to output CSV file",
    )
    ap.add_argument(
        "--num_ranks",
        "-np",
        type=int,
        required=True,
        help="Number of ranks used to run the program",
    )
    ap.add_argument(
        "--num_nodes", "-n", type=int, required=True, help="Number of nodes allocated"
    )
    ap.add_argument(
        "--with_hints",
        type=str,
        default="romio",
        help="MPIIO hints mode",
        choices=["romio", "cray", "ompio"],
    )
    ap.add_argument(
        "-b",
        "--tune_bandwidth",
        action="store_true",
        default=False,
        help="Use I/O bandwidth as the tuning objective",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent.parent / "configs" / "tune_config_romio.json",
        help="Path to JSON configuration file (default: ../configs/tune_config_romio.json",
    )
    ap.add_argument("--nsamples", type=int, default=50, help="Number of samples")

    ap.add_argument(
        "--with_log_path", type=str, default=None, help="Output logging path"
    )
    ap.add_argument(
        "--cmd",
        "-c",
        type=str,
        required=True,
        nargs="*",
        help="Application command line",
    )
    args = vars(ap.parse_args(args))

    global num_ranks, num_nodes, ioflexset, run_app, outfilepath, outfile, logisset, logfile_o, logfile_e, hints, tune_bandwidth, files_to_clean, files_to_stripe
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    tune_bandwidth = args["tune_bandwidth"]

    outfilepath = args["outfile"]
    global out_is_open
    out_is_open = False

    outdir = os.path.dirname(os.path.abspath(outfilepath))

    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]
    nsamples = args["nsamples"]

    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    # Define configurations mappings
    hints = args["with_hints"]
    config_path = args["config"]

    CONFIG_MAP, files_to_clean, files_to_stripe = get_config_map(hints, config_path)
    config_space = {key: value for key, value in sorted(CONFIG_MAP.items()) if value}


    lhs_samples = generate_lhs_samples(config_space, nsamples)
    # Run with Samples
    eval_runs(lhs_samples, nsamples)

    if out_is_open:
        outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()
