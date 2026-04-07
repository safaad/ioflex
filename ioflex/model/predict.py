import pandas as pd
import argparse
from pathlib import Path
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
import json
from sklearn.model_selection import ParameterGrid
from ioflex.common import (
    get_config_map,
    are_cray_hints_valid,
)


# Generate All Valid Combinations of this Configuration Space
def generate_parameter_grid(config_space, num_nodes, num_ranks, hints):

    config_space["nodes"] = [num_nodes]
    grid = ParameterGrid(config_space)
    # eliminate non-valid options
    if hints == "cray":
        valid_records = [
            p for p in grid if are_cray_hints_valid(p, num_ranks, num_nodes)
        ]
        dfall = pd.DataFrame(valid_records)
    else:
        dfall = pd.DataFrame(grid)

    return dfall


def preprocess_data(df, features_in):
    dfnew = pd.DataFrame()
    
    numerics = [ "ratio", "nodes", "cb_nodes", "cray_cb_nodes_multiplier", "striping_factor" ] 
    cats = [ "romio_cb_read", "romio_cb_write", "romio_ds_read", "romio_ds_write"]
    cats_binary = ["romio_no_indep_rw"]
    ordinal = ["cray_cb_write_lock_mode", "striping_unit"]

    for c in ordinal:
        if c in df.columns:
            df[c] = df[c].astype('category')
            df[c] = df[c].cat.codes
    for c in cats_binary:
        if c in df.columns:
            df[c] = df[c].astype('bool')

    df_encoded = pd.get_dummies(df, columns=cats, drop_first=True)

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')

    one_hot_encoded = enc.fit_transform(df.loc[:, cats])

    one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=enc.get_feature_names_out(cats))

    one_hot_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_encoded = pd.concat([df.drop(cats, axis=1),one_hot_df], axis=1)
    
    # Order columns as trained
    df_encoded = df_encoded[features_in]
    return df_encoded


def run(args=None):

    ap = argparse.ArgumentParser(add_help=True)
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
        "--config",
        type=str,
        default=Path(__file__).parent.parent / "configs" / "tune_config_romio.json",
        help="Path to JSON configuration file (default: ../configs/tune_config_romio.json",
    )
    ap.add_argument(
        "--with_hints",
        type=str,
        default="romio",
        help="MPIIO hints mode",
        choices=["romio", "cray", "ompio"],
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./OutPredictedGrid.csv",
        help="Path to output CSV file",
    )
    ap.add_argument("--model", type=str, help="Path to the trained model")
    args = vars(ap.parse_args(args))

    # Define configurations mappings
    hints = args["with_hints"]
    config_path = args["config"]

    global num_ranks, num_nodes

    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]

    CONFIG_MAP, files_to_clean, files_to_stripe = get_config_map(hints, config_path)
    config_space = {key: value for key, value in sorted(CONFIG_MAP.items()) if value}

    dfgrid = generate_parameter_grid(config_space, num_nodes, num_ranks, hints)

    model = joblib.load(args["model"])
    
    X = preprocess_data(dfgrid, model.feature_names_in_)
    y_hat = model.predict(X)


    dfgrid["predicted"] = y_hat
    
    best_predicted_config = dfgrid.loc[dfgrid['predicted'].idxmax()]

    # Convert to dictionary
    best_predicted_config = best_predicted_config.to_dict()
    print(f"Best Configuration Predicted: {best_predicted_config}")
    
    
    outfilepath = args["outfile"]
    outdir = os.path.dirname(os.path.abspath(outfilepath))
    
    dfgrid.to_csv(outfilepath)
    
    bestpath = os.path.join(
        outdir, os.path.basename(outfilepath).split(".")[0] + "_best_predicted_params.json"
    )
    with open(bestpath, "w") as f:
        json.dump(best_predicted_config, f, indent=4)