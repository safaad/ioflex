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

import pygad
import numpy


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


# GA parameters
iter_count = 0
gene_space = []



# Application Specific 
# List of files to clean after running application
files_to_clean = []

def eval_func(ga_instance, solution, solution_idx):

    dir_path = os.environ['PWD']

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
    
    # print("Running with configs: " + configs_str)

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
    fitness = 1.0 / (elapsedtime)
    return fitness

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def ioflexgad():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--ioflex', action="store_true", default=False,help="IOFlex is enabled")

    ## TODO
    ap.add_argument('--outfile', type=str, help="Path to output history runs file", default="./OutIOFlexGA.csv")
    ap.add_argument('--infile', type=str, help="Path to file from previous runs", default=None)
    ap.add_argument('--outgainstance', '-o', type=str, help="Genetic algorithm instance output file", default=None)
    ap.add_argument('--cmd', '-c', type=str, required=True, action="store", nargs="*", help="Application command line")
    # Genetic Algorithm Options
    ap.add_argument("--select_type", type=str, help="GA parent selection type, default=rws",default="rws" )
    ap.add_argument("--cross_prob", type=float, help="crossover probability values between 0.0 and 1.0", default=0.2)
    ap.add_argument("--mut_prob", type=float, help="mutation probability values between 0.0 and 1.0", default=0.2)
    ap.add_argument("--num_gens",type=int, help="Number of generations", default=40)
    ap.add_argument("--num_parents", type=int, help="Number of parents mating", default=5)
    ap.add_argument("--pop_size", type=int, help="Population size", default=10)
    ap.add_argument("--cross_type", type=str, help="crossover type", default="scattered")
    ap.add_argument("--elite-size", type=int, help="Elite size", default=4)
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


    print("Starting IOFlex Genetic Algorithm")

    global num_generations, pop_size, elite_size, iter_count


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

    if (len(striping_factor)):
        gene_space.append(striping_factor)
        strf_id = no_of_configs
        no_of_configs += 1
    if (len(striping_unit)):
        gene_space.append(striping_unit)
        stru_id = no_of_configs 
        no_of_configs += 1
    if (len(cb_nodes)):
        gene_space.append(cb_nodes)
        cbn_id = no_of_configs
        no_of_configs += 1
    if (len(romio_fs_type)):
        gene_space.append([romio_fs_type.index(x) for x in romio_fs_type])
        fstype_id = no_of_configs
        no_of_configs += 1
    if (len(romio_ds_read)):
        gene_space.append([romio_ds_read.index(x) for x in romio_ds_read])
        dsread_id = no_of_configs
        no_of_configs += 1
    if (len(romio_ds_write)):
        gene_space.append([romio_ds_write.index(x) for x in romio_ds_write])
        dswrite_id = no_of_configs
        no_of_configs += 1
    if (len(romio_cb_read)):
        gene_space.append([romio_cb_read.index(x) for x in romio_cb_read])
        cbread_id = no_of_configs
        no_of_configs += 1
    if (len(romio_cb_write)):
        gene_space.append([romio_cb_write.index(x) for x in romio_cb_write])
        cbwrite_id  = no_of_configs
        no_of_configs += 1 
    if (len(cb_config_list)):
        gene_space.append([cb_config_list.index(x) for x in cb_config_list])
        cbl_id = no_of_configs
        no_of_configs += 1
    

    global num_generations, pop_size, elite_size
    num_generations = args["num_gens"]
    pop_size = args["pop_size"]
    elite_size = args["elite_size"]
    num_parents_mating = args["num_parents"]
    crossover_type = args["cross_type"]
    parent_selection_type = args["select_type"]
    crossover_probability = args["cross_prob"]
    mutation_probability = args["mut_prob"]
    sol_per_pop = len(gene_space)
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=no_of_configs,
                        gene_space=gene_space,
                        gene_type=int,
                        fitness_func=eval_func,
                        parent_selection_type=parent_selection_type,
                        keep_elitism=elite_size,
                        crossover_type=crossover_type,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability,
                        suppress_warnings=True,
                        random_seed=123,
                        save_solutions=True,
                        on_generation=on_generation)
    print("Run Evolve")
    ga_instance.run()


    print("Saving Results")

    if args["outgainstance"] is not None:
        ga_instance.save(filename=args["outgainstance"])


    outfile.close()
    if infile is not None:
        infile.close()
        
    # print(f"Solution : {ga_instance.solutions}")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")


    # ga_instance.plot_fitness(plot_type="bar", color="red", save_dir=".")
    # ga_instance.plot_genes(graph_type="plot", 
    #                     plot_type="bar", 
    #                     solutions='all')

if __name__ == '__main__':
    ioflexgad()
