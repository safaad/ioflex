import os
import sys
import time
import argparse
import shlex
import subprocess
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from shutil import rmtree
from scipy.stats import qmc
import parsedarshan
import glob


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = configure_logging()

CONFIG_ENTRIES = {
    "striping_factor": [4, 8, 16, 24, 32, 40, 48, 64, -1],
    "striping_unit": [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728],
    "cb_nodes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "romio_fs_type": ["LUSTRE:", "UFS:"],
    "romio_ds_read": ["automatic", "enable", "disable"],
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"]
}


files_to_clean = []

def generate_lhs_samples(param_grid, nsamples):
    n_dims = len(param_grid)
    lhs = qmc.LatinHypercube(d=n_dims)
    lhs_samples = lhs.random(n=nsamples)
    
    sampled_configs = []
    for i in range(nsamples):
        sampled_config = []
        for j, param_values in enumerate(param_grid):
            index = int(np.floor(lhs_samples[i, j] * len(param_values)))
            sampled_config.append(param_values[index])
        sampled_configs.append(sampled_config)
    
    return sampled_configs


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--ioflex', action="store_true", default=False, help="Enable IOFlex")
    ap.add_argument('--outfile', type=str, default="./OutIOAllRuns.csv", help="Path to output history runs file")
    ap.add_argument('--app_name', type=str, required=True, help="Application Name")
    ap.add_argument('--num_nodes', type=int, required=True, help="Number of nodes")
    ap.add_argument('--nsamples', type=int, default=50, help="Number of samples")
    ap.add_argument('--mpi_procs', type=int, required=True, help="Number of MPI processes per node")
    ap.add_argument('--darshan_path', type=str, help="Path to Darshan output")
    ap.add_argument('--cmd', '-c', type=str, required=True, nargs="*", help="Application command line")
    
    args = vars(ap.parse_args())
    global ioflexset, run_app, outfile
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], 'w')
    
    sampled_configs = generate_lhs_samples(list(CONFIG_ENTRIES.values()), args["nsamples"])
    logger.info(f"Number of samples: {len(sampled_configs)}")
    
    df = pd.DataFrame()
    for solution in sampled_configs:
        iodict = {'appname': args["app_name"], 'num_nodes': args["num_nodes"], 'mpi_procs': args["mpi_procs"]}
        
        config_path = os.path.join(os.getcwd(), "config.conf" if ioflexset else "romio-hints")
        with open(config_path, "w") as config_file:
            config_entries = []
            for key, values in CONFIG_ENTRIES.items():
                if values:
                    selected_val = solution[list(CONFIG_ENTRIES.keys()).index(key)]
                    separator = " = " if ioflexset else " "
                    
                    config_file.write(f"{key}{separator}{selected_val}\n")
                    config_entries.append(str(selected_val))

                if key == "striping_unit":
                    config_file.write(f"cb_buffer_size{separator}{selected_val}\n")
                elif key == "romio_filesystem_type":
                    os.environ["ROMIO_FSTYPE_FORCE"] = selected_val 

        configs_str = ",".join(config_entries)
        if not ioflexset:
            os.environ["ROMIO_HINTS"] = config_path
        
        starttime = time.time()
        process = subprocess.Popen(shlex.split(run_app), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.communicate()
        elapsedtime = time.time() - starttime
        iodict["runtime"] = elapsedtime
        
        logfile = os.path.join(args["darshan_path"], "*.darshan")
        for f in glob.glob(logfile):
            dictdarshan = parsedarshan.parsedarshan(f)
            dictdarshan = parsedarshan.merge(iodict, dictdarshan)
            df = pd.concat([df, pd.DataFrame([dictdarshan])], ignore_index=True)
            os.remove(f)
        
        df.to_csv('parsing.csv', index=False)
        
        for f in files_to_clean:
            if os.path.exists(f):
                os.remove(f) if os.path.isfile(f) else rmtree(f)
    
    outfile.close()

if __name__ == '__main__':
    main()