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

import numpy
import itertools

# Set of IO Configurations
## Lustre Striping
striping_factor = [4, 8, 16, 24, 32, 40, 48, 64, -1]
striping_unit = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728]


## Collective Buffering
### Adjust number of cb_nodes to total number of nodes
cb_nodes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
cb_config_list = []

## ROMIO Optimizations
romio_fs_type = ["\"LUSTRE:\"", "\"UFS:\""]
romio_ds_read = ["automatic", "enable", "disable"]
romio_ds_write = ["automatic", "enable", "disable"]
romio_cb_read = ["automatic", "enable", "disable"]
romio_cb_write = ["automatic", "enable", "disable"]


iter_count = 0
files_to_clean = []

def runall():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--ioflex', action="store_true", default=False,help="IOFlex is enabled")
    ap.add_argument('--outfile', type=str, help="Path to output history runs file", default="./OutIOAllRuns.csv")
    ap.add_argument('--infile', type=str, help="Path to file from previous runs", default=None)
    ap.add_argument('--cmd', '-c', type=str, required=True, action="store", nargs="*", help="Application command line")

    parse = ap.parse_args()
    parse = ap.parse_args()
    args = vars(parse)
    global ioflexset
    ioflexset = args["ioflex"]

    global run_app
    run_app = " ".join(args["cmd"])

    ## TODO
    global infile, outfile
    if args["infile"] is not None:
        infile = open(args["infile"],'r')
    else:
        infile = None
    
    global outfile
    outfile = open(args["outfile"], 'w')

    global strf_id, stru_id, cbn_id, fstype_id, dsread_id, dswrite_id, cbread_id, cbwrite_id, cbl_id
    no_of_configs = 0
    strf_id = None
    stru_id = None
    cbn_id = None
    fstype_id = None
    dsread_id = None
    dswrite_id = None
    cbread_id = None
    cbwrite_id = None
    cbl_id = None
    all_configs = []

    if (len(striping_factor)):
        strf_id = no_of_configs
        no_of_configs += 1
        all_configs.append(striping_factor)
    if (len(striping_unit)):
        stru_id = no_of_configs 
        no_of_configs += 1
        all_configs.append(striping_unit)
    if (len(cb_nodes)):
        cbn_id = no_of_configs
        no_of_configs += 1
        all_configs.append(cb_nodes)
    if (len(romio_fs_type)):
        fstype_id = no_of_configs
        no_of_configs += 1
        all_configs.append([romio_fs_type.index(x) for x in romio_fs_type])
    if (len(romio_ds_read)):
        dsread_id = no_of_configs
        no_of_configs += 1
        all_configs.append([romio_ds_read.index(x) for x in romio_ds_read])
    if (len(romio_ds_write)):
        dswrite_id = no_of_configs
        no_of_configs += 1
        all_configs.append([romio_ds_write.index(x) for x in romio_ds_write])
    if (len(romio_cb_read)):
        cbread_id = no_of_configs
        no_of_configs += 1
        all_configs.append([romio_cb_read.index(x) for x in romio_cb_read])
    if (len(romio_cb_write)):
        cbwrite_id  = no_of_configs
        no_of_configs += 1
        all_configs.append([romio_cb_write.index(x) for x in romio_cb_write]) 
    if (len(cb_config_list)):
        cbl_id = no_of_configs
        no_of_configs += 1
        all_configs.append([cb_config_list.index(x) for x in cb_config_list])

    dir_path = os.environ['PWD']



    all_solutions = list(itertools.product(*all_configs))
    print("Number of all configs: " + str(len(all_solutions)) + '\n')

    for solution in all_solutions:
        # Set IO Hints using IOFlex or ROMIO HINTS
        if ioflexset:
            config_file = open(os.path.join(dir_path,"config.conf"),'w')
        else:
            romio_path = os.path.join(dir_path,"romio-hints")
            config_file = open(romio_path,'w')
        # Set IO Configurations
        configs_str=""
        if strf_id is not None:
            strf_val = solution[strf_id]
            if ioflexset:
                config_file.write("striping_factor = " + str(strf_val)+"\n")
            else:
                config_file.write("striping_factor " + str(strf_val)+"\n")
            configs_str += "striping_factor," + str(strf_val) + ","
        if stru_id is not None:
            stru_val = solution[stru_id]
            if ioflexset:
                config_file.write("striping_unit = " + str(stru_val)+"\n")
            else:
                config_file.write("striping_unit " + str(stru_val)+"\n")
            configs_str += "striping_unit," + str(stru_val) + ","
        if cbn_id is not None:
            cbn_val = solution[cbn_id]
            if ioflexset:
                config_file.write("cb_nodes = " + str(cbn_val)+"\n")
            else:
                config_file.write("cb_nodes " + str(cbn_val)+"\n")
            configs_str += "cb_nodes," + str(cbn_val) + ","
        if fstype_id is not None:
            fstype_val = romio_fs_type[solution[fstype_id]]
            if ioflexset:
                config_file.write("romio_filesystem_type = " + fstype_val+"\n")
            else:
                config_file.write("romio_filesystem_type " + fstype_val+"\n")
            configs_str += "romio_filesystem_type," + fstype_val + ","
        if dsread_id is not None:
            dsread_val = romio_ds_read[solution[dsread_id]]
            if ioflexset:
                config_file.write("romio_ds_read = " + dsread_val+"\n")
            else:
                config_file.write("romio_ds_read " + dsread_val+"\n")
            configs_str += "romio_ds_read," + dsread_val + ","
        if dswrite_id is not None:
            dswrite_val = romio_ds_write[solution[dswrite_id]]
            if ioflexset:
                config_file.write("romio_ds_write = " + dswrite_val+"\n")
            else:
                config_file.write("romio_ds_write " + dswrite_val+"\n")
            configs_str += "romio_ds_write," + dswrite_val + ","
        if cbread_id is not None:
            cbread_val = romio_cb_read[solution[cbread_id]]
            if ioflexset:
                config_file.write("romio_cb_read = " + cbread_val+"\n")
            else:
                config_file.write("romio_cb_read " + cbread_val+"\n")
            configs_str += "romio_cb_read," + cbread_val + ","
        if cbwrite_id is not None:
            cbwrite_val = romio_cb_write[solution[cbwrite_id]]
            if ioflexset:
                config_file.write("romio_cb_write = " + cbwrite_val+"\n")
            else:
                config_file.write("romio_cb_write " + cbwrite_val +"\n")
            configs_str += "romio_cb_write," + cbwrite_val + ","
        if cbl_id is not None:
            cbl_val = cb_config_list[solution[cbl_id]]
            if ioflexset:
                config_file.write("cb_config_list = " + cbl_val+"\n")
            else:
                config_file.write("cb_config_list " + cbl_val+"\n")
            configs_str += "cb_config_list," + cbl_val + ","
        config_file.close()


        # Use ROMIO HINTS if IOFLex is not enabled
        if not ioflexset:
            os.environ['ROMIO_HINTS'] = romio_path

        config_exist = None
        # check if config already exists
        if infile is not None:
            config_exist = [line for line in infile if configs_str in line]
        
        if config_exist is not None and len(config_exist)!= 0:
            elapsedtime = float(config_exist[0].split(',elapsedtime,')[1])
            outline = configs_str + "elapsedtime," + str(elapsedtime)+"\n"
            outfile.write(outline)
            print("Config found:" + outline)
            return elapsedtime

        # start application
        starttime = 0.0
        elapsedtime = 0.0

        starttime = time.time()
        q = subprocess.Popen(run_app, stdout=subprocess.PIPE, shell=True)
        out, err = q.communicate()
        elapsedtime = time.time() - starttime
        # Write output of this config
        outline = configs_str + "elapsedtime," + str(elapsedtime)+"\n"
        outfile.write(outline)
        
        # Clean application files after every iteration
        for f in files_to_clean:
            if os.path.isfile(f):
                os.remove(f)
            if os.path.isdir(f):
                rmtree(f)
        print("Running config: " + outline)
    outfile.close()
    if infile is not None:
        infile.close()

if __name__ == '__main__':
    runall()