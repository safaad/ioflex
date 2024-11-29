import pandas as pd
import darshan
import numpy as np
import glob
import os
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import argparse
import pickle
#import  utils

time_cols = [
    "MPIIO_F_READ_TIME",
    "MPIIO_F_WRITE_TIME",
    "MPIIO_F_META_TIME",
    "POSIX_F_READ_TIME",
    "POSIX_F_WRITE_TIME",
    "POSIX_F_META_TIME",
]
mpiio_acc = [
    "MPIIO_SIZE_READ_AGG_0_100",
    "MPIIO_SIZE_WRITE_AGG_0_100",
    "MPIIO_SIZE_READ_AGG_100_1K",
    "MPIIO_SIZE_WRITE_AGG_100_1K",
    "MPIIO_SIZE_READ_AGG_1K_10K",
    "MPIIO_SIZE_WRITE_AGG_1K_10K",
    "MPIIO_SIZE_READ_AGG_10K_100K",
    "MPIIO_SIZE_WRITE_AGG_10K_100K",
    "MPIIO_SIZE_READ_AGG_100K_1M",
    "MPIIO_SIZE_WRITE_AGG_100K_1M",
    "MPIIO_SIZE_READ_AGG_1M_4M",
    "MPIIO_SIZE_WRITE_AGG_1M_4M",
    "MPIIO_SIZE_READ_AGG_4M_10M",
    "MPIIO_SIZE_WRITE_AGG_4M_10M",
    "MPIIO_SIZE_READ_AGG_10M_100M",
    "MPIIO_SIZE_WRITE_AGG_10M_100M",
    "MPIIO_SIZE_READ_AGG_100M_1G",
    "MPIIO_SIZE_WRITE_AGG_100M_1G",
    "MPIIO_SIZE_READ_AGG_1G_PLUS",
    "MPIIO_SIZE_WRITE_AGG_1G_PLUS",
]
mpiio_ops = [
    "MPIIO_INDEP_OPENS",
    "MPIIO_COLL_OPENS",
    "MPIIO_INDEP_WRITES",
    "MPIIO_COLL_WRITES",
    "MPIIO_INDEP_READS",
    "MPIIO_COLL_READS",
    "MPIIO_SYNCS",
]
mpiio_bytes = ["MPIIO_BYTES_READ", "MPIIO_BYTES_WRITTEN"]

posix_acc = [
    "POSIX_CONSEC_READS",
    "POSIX_CONSEC_WRITES",
    "POSIX_SEQ_READS",
    "POSIX_SEQ_WRITES",
    "POSIX_SIZE_READ_0_100",
    "POSIX_SIZE_WRITE_0_100",
    "POSIX_SIZE_READ_100_1K",
    "POSIX_SIZE_WRITE_100_1K",
    "POSIX_SIZE_READ_1K_10K",
    "POSIX_SIZE_WRITE_1K_10K",
    "POSIX_SIZE_READ_10K_100K",
    "POSIX_SIZE_WRITE_10K_100K",
    "POSIX_SIZE_READ_100K_1M",
    "POSIX_SIZE_WRITE_100K_1M",
    "POSIX_SIZE_READ_1M_4M",
    "POSIX_SIZE_WRITE_1M_4M",
    "POSIX_SIZE_READ_4M_10M",
    "POSIX_SIZE_WRITE_4M_10M",
    "POSIX_SIZE_READ_10M_100M",
    "POSIX_SIZE_WRITE_10M_100M",
    "POSIX_SIZE_READ_100M_1G",
    "POSIX_SIZE_WRITE_100M_1G",
    "POSIX_SIZE_READ_1G_PLUS",
    "POSIX_SIZE_WRITE_1G_PLUS",
]

posix_ops = [
    "POSIX_STATS",
    "POSIX_SEEKS",
    "POSIX_MMAPS",
    "POSIX_FSYNCS",
    "POSIX_WRITES",
    "POSIX_READS",
]
posix_bytes = ["POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN"]

files_col = ["nb_read-only_files", "nb_write-only_files", "nb_read-write_files"]

categ_cols = [
    "romio_filesystem_type",
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
]
discr_cols = ["striping_factor", "striping_unit"]

config_cols = [
    "romio_filesystem_type",
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
    "striping_factor",
    "striping_unit",
]


def smape(y_true, y_pred):
    """
    Calculate SMAPE (Symmetric Mean Absolute Percentage Error)

    Parameters:
    y_true : array-like of shape (n_samples,)
        True values of the target variable
    y_pred : array-like of shape (n_samples,)
        Predicted values of the target variable

    Returns:
    float
        SMAPE score as a percentage
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean(numerator / denominator) * 100
    return smape_value


def read_dataset(filepath):
    all_files = glob.glob(os.path.join(filepath, "*.csv"))
    dfall = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


def normalize_dataset(dfall):
    total_posix_bytes = dfall.POSIX_BYTES_READ + dfall.POSIX_BYTES_WRITTEN
    total_posix_ops = (
        dfall.POSIX_WRITES
        + dfall.POSIX_READS
        + dfall.POSIX_STATS
        + dfall.POSIX_SEEKS
        + dfall.POSIX_MMAPS
        + dfall.POSIX_FSYNCS
    )
    total_posix_accesses = dfall.POSIX_WRITES + dfall.POSIX_READS

    total_mpiio_bytes = dfall.MPIIO_BYTES_READ + dfall.MPIIO_BYTES_WRITTEN
    total_mpiio_accesses = (
        dfall.MPIIO_INDEP_WRITES
        + dfall.MPIIO_COLL_WRITES
        + dfall.MPIIO_INDEP_READS
        + dfall.MPIIO_COLL_READS
    )
    total_mpiio_ops = (
        dfall.MPIIO_INDEP_OPENS
        + dfall.MPIIO_COLL_OPENS
        + dfall.MPIIO_INDEP_WRITES
        + dfall.MPIIO_COLL_WRITES
        + dfall.MPIIO_INDEP_READS
        + dfall.MPIIO_COLL_READS
        + dfall.MPIIO_SYNCS
    )

    dfall["POSIX_BYTES_TOTAL"] = total_posix_bytes
    dfall["POSIX_OPS"] = total_posix_ops
    dfall["POSIX_ACCESS_TOTAL"] = total_posix_accesses

    dfall["MPIIO_BYTES_TOTAL"] = total_mpiio_bytes
    dfall["MPIIO_OPS"] = total_mpiio_ops
    dfall["MPIIO_ACCESS_TOTAL"] = total_mpiio_accesses

    df = pd.DataFrame()
    df = pd.DataFrame()
    for c in time_cols:
        df[c] = dfall[c] / dfall["runtime"]
    for c in mpiio_acc:
        df[c] = dfall[c] / dfall["MPIIO_ACCESS_TOTAL"]
    for c in mpiio_bytes:
        df[c] = dfall[c] / dfall["MPIIO_BYTES_TOTAL"]
    for c in mpiio_ops:
        df[c] = dfall[c] / dfall["MPIIO_OPS"]

    for c in posix_acc:
        df[c] = dfall[c] / dfall["POSIX_ACCESS_TOTAL"]
    for c in posix_ops:
        df[c] = dfall[c] / dfall["POSIX_OPS"]
    for c in posix_bytes:
        df[c] = dfall[c] / dfall["POSIX_BYTES_TOTAL"]

    for c in files_col:
        df[c] = dfall[c] / dfall["total_files"]

    for c in categ_cols:
        dfall[c] = dfall[c].astype("category")
        df[c] = dfall[c].cat.codes

    for c in discr_cols:
        df[c] = dfall[c]

    df["runtime"] = dfall["runtime"]
    return df


def get_data(df):
    df = df.drop("config",axis=1)
    X = df.loc[:, (df.columns != "runtime")].to_numpy()
    y = df["runtime"].to_numpy()
    return X, y


def split_train_test(df):
    # make sure to remove overlapping values
    df["config"] = (
        df["romio_filesystem_type"].astype(str)
        + "-"
        + df["romio_ds_read"].astype(str)
        + "-"
        + df["romio_ds_write"].astype(str)
        + "-"
        + df["romio_cb_read"].astype(str)
        + "-"
        + df["romio_cb_write"].astype(str)
        + "-"
        + df["striping_factor"].astype(str)
        + "-"
        + df["striping_unit"].astype(str)
    )
    train_conf, test_conf = train_test_split(
        df["config"].unique(), random_state=123, test_size=0.2, shuffle=True
    )
    train = df.loc[df["config"].isin(train_conf)]
    test = df.loc[df["config"].isin(test_conf)]
    
    train = train.drop("config", axis=1)
    test = test.drop("config", axis=1)

    X_train = train.loc[:, (train.columns != "runtime")].to_numpy()
    X_test = test.loc[:, (test.columns != "runtime")].to_numpy()
    y_train = train["runtime"].to_numpy()
    y_test = test["runtime"].to_numpy()
    return X_train, X_test, y_train, y_test


def create_test_xgboost_model(X_train, X_test, y_train, y_test):
    # Initialize XGBoost Regressor
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )

    # Train the model
    xg_reg.fit(X_train, y_train)

    # Predict on test set
    y_pred = xg_reg.predict(X_test)

    # Calculate SMAPE
    smape_value = smape(y_test, y_pred)
    print(f"SMAPE: {smape_value:.2f}%")
    return xg_reg


def create_xgboost_model(X, y):
    # Initialize XGBoost Regressor
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    # Train the model
    xg_reg.fit(X, y)
    return xg_reg

def save_model(outmodelfile, model):
    with open(outmodelfile, 'wb') as file:  
        pickle.dump(model, file)

def load_model(modelfile):
    with open(modelfile, 'rb') as file:  
        model = pickle.load(file)
        return model


def ioflexpredict():
    #printheader()
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--algo",
        type=str,
        default="xgb",
        help="Machine Learning Algorithm",
        choices=["xgb"]
    )
    ap.add_argument(
        "--outmodelfile",
        type=str,
        help="Filename to save the pickeled  model",
        default="model.pkl"
    )
    ap.add_argument(
        "--dataset",
        type=str,
        help="Path to the training dataset",
        required=True
    )
    ap.add_argument("--test",
                    action="store_true",
                    default=False,
                    help="Get initial score of mode")
    parse = ap.parse_args()
    args = vars(parse)
    
    algo = args["algo"]
    dataset = args["dataset"]
    outmodelfile = args["outmodelfile"]
    
    dfall = pd.read_csv(dataset)
    df = normalize_dataset(dfall)
    
    # initial model score
    if args["test"]:
        X_train, X_test, y_train, y_test = split_train_test(df)
        create_test_xgboost_model(X_train, X_test, y_train, y_test)
    
    # Train Model
    X, y = get_data(df)
    match algo:
        case "xgb":
            model = create_xgboost_model(X,y)
    
    # Pickle and save file
    save_model(outmodelfile, model)    
    
    

if __name__ == "__main__":
    ioflexpredict()
