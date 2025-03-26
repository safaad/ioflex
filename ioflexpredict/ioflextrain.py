import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import argparse
import optuna
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import itertools
# import  utils
import joblib

config_cols = [
    "romio_filesystem_type",
    "romio_ds_read",
    "romio_ds_write",
    "romio_cb_read",
    "romio_cb_write",
    "striping_factor",
    "striping_unit",
    "cb_nodes",
    "cb_config_list",
]

# Lustre Striping
striping_factor = [4, 8, 16, 24, 32, 40, 48]
striping_unit = [1048576, 2097152, 4194304, 8388608]

## Collective Buffering
### Adjust number of cb_nodes to total number of nodes
cb_nodes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
cb_config_list = ["*:1", "*:2", "*:4", "*:8", "*:32", "*:64", "*:128"]

## ROMIO Optimizations
romio_filesystem_type = ["LUSTRE:", "UFS:"]
romio_ds_read = ["automatic", "enable", "disable"]
romio_ds_write = ["automatic", "enable", "disable"]
romio_cb_read = ["automatic", "enable", "disable"]
romio_cb_write = ["automatic", "enable", "disable"]

# Define possible values for categorical features
feature_values = {
    "striping_unit": [
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
    ],
    "cb_config_list": ["*:1", "*:2", "*:4", "*:8", "*:16", "*:32", "*:64", "*:128"],
    "romio_filesystem_type": ["LUSTRE:", "UFS:"],
    "romio_ds_read": ["automatic", "enable", "disable"],
    "romio_ds_write": ["automatic", "enable", "disable"],
    "romio_cb_read": ["automatic", "enable", "disable"],
    "romio_cb_write": ["automatic", "enable", "disable"],
}

category_mappings = {
    feature: {category: idx for idx, category in enumerate(values)}
    for feature, values in feature_values.items()
}
print(category_mappings)

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


def random_forest_regression_optuna(X, y, n_trials=100):
    """
    Perform hyperparameter tuning for Random Forest Regression using Optuna.

    Args:
    - X: Features (input data)
    - y: Target values (labels)
    - n_trials: Number of trials to run in Optuna (default: 100)

    Returns:
    - Best trial parameters
    """

    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Objective function for Random Forest tuning with SMAPE minimization
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=200),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,  # Ensure reproducibility
        }

        # Train the model with suggested hyperparameters
        model = RandomForestRegressor(**param)
        model.fit(X_train, y_train)

        # Predict and calculate SMAPE
        preds = model.predict(X_valid)

        # Calculate SMAPE as the evaluation metric
        smape_score = smape(y_valid, preds)
        return smape_score  # Return SMAPE to minimize it

    sampler = optuna.samplers.GPSampler()
    # Create Optuna study to minimize SMAPE
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Optimize the model
    study.optimize(objective, n_trials=n_trials)
    print(f"Best Trial: {study.best_trial.params}")

    # Output the best trial parameters
    return study.best_trial.params


def dt_regression_optuna(X, y, n_trials=100):
    """
    Perform hyperparameter tuning for Decision Tree Regression using Optuna.

    Args:
    - X: Features (input data)
    - y: Target values (labels)
    - n_trials: Number of trials to run in Optuna (default: 100)

    Returns:
    - Best trial parameters
    """

    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Objective function for Decision Tree tuning with SMAPE minimization
    def objective(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error"]
            ),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        }

        # Train the model with suggested hyperparameters
        model = DecisionTreeRegressor(**param, random_state=42)
        model.fit(X_train, y_train)

        # Predict and calculate SMAPE
        preds = model.predict(X_valid)

        # Calculate SMAPE as the evaluation metric
        smape_score = smape(y_valid, preds)
        return smape_score  # Return SMAPE to minimize it

    # Create Optuna study to minimize SMAPE
    study = optuna.create_study(direction="minimize")

    # Optimize the model
    study.optimize(objective, n_trials=n_trials)
    print(f"Best Trial: {study.best_trial.params}")

    # Output the best trial parameters
    return study.best_trial.params


def xgb_regression_optuna(X, y, n_trials=100):

    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Objective function for XGBoost tuning with SMAPE minimization
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            'random_state': 42
        }

        # Train the model with suggested hyperparameters
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)

        # Predict and calculate SMAPE
        preds = model.predict(X_valid)

        smape_score = smape(y_valid, preds)
        return smape_score

    # Create Optuna study to minimize SMAPE
    study = optuna.create_study(direction="minimize")

    # Optimize the model
    study.optimize(objective, n_trials=n_trials)
    print(f"Best Trial: {study.best_trial.params}")

    # Output the best trial parameters
    return study.best_trial.params

def xgb_regression_gridsearch(X, y):

    # Define hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [50, 100, 200, 500, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # Initialize XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Perform GridSearchCV for hyperparameter tuning
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, 
                            scoring='neg_mean_squared_error', cv=kf, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    return best_params
        
def preprocess_dataset(dfall):

    df = pd.DataFrame()
    for col in category_mappings:
        df[col] = dfall[col].map(category_mappings[col]).fillna(-1).astype(int)
    df['striping_factor'] = dfall['striping_factor'].astype(int)
    df['cb_nodes'] = dfall['cb_nodes'].astype(int)
    
    df["runtime"] = dfall["runtime"]
    return df


def get_data(df):
    df = df.drop("config", axis=1)
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
        df["config"].unique(), random_state=123, test_size=0.25, shuffle=True
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


def test_model(X_train, X_test, y_train, y_test, best_params, algo):

    match algo:
        case "rf":
            model = RandomForestRegressor(**best_params, random_state=42)
        case "dt":
            model = DecisionTreeRegressor(**best_params, random_state=42)
        case _:
            model = xgb.XGBRegressor(**best_params, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    smape_value = smape(y_test, y_pred)
    print(f"SMAPE: {smape_value:.2f}%")
    return smape_value


def load_model(modelfile):
    with open(modelfile, "rb") as file:
        model = pickle.load(file)
        return model


def predictallcombinations(model):

    combinations = list(
        itertools.product(
            striping_factor,
            striping_unit,
            cb_nodes,
            cb_config_list,
            romio_filesystem_type,
            romio_ds_read,
            romio_ds_write,
            romio_cb_read,
            romio_cb_write,
        )
    )

    # Create a Pandas DataFrame
    df = pd.DataFrame(
        combinations,
        columns=[
            "striping_factor",
            "striping_unit",
            "cb_nodes",
            "cb_config_list",
            "romio_filesystem_type",
            "romio_ds_read",
            "romio_ds_write",
            "romio_cb_read",
            "romio_cb_write",
        ],
    )
    test = pd.DataFrame()
    for c in config_cols:
        df[c] = df[c].astype("category")
        test[c] = df[c].cat.codes

    X_test = test.to_numpy()
    y_test = model.predict(X_test)

    df["pred-time"] = y_test
    df.to_csv("predicted.csv")


def ioflextrain():
    # printheader()
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--algo",
        type=str,
        default="xgb",
        help="Machine Learning Algorithm",
        choices=["xgb", "rf", "dt"],
    )
    ap.add_argument(
        "--outmodelfile",
        type=str,
        help="Filename to save the pickeled  model",
        default="model.pkl",
    )
    ap.add_argument(
        "--dataset", type=str, help="Path to the training dataset", required=True
    )
    ap.add_argument(
        "--ntrials",
        type=int,
        default=50,
        help="Number of trials to run hyperparametertuning",
    )
    parse = ap.parse_args()
    args = vars(parse)

    algo = args["algo"]
    dataset = args["dataset"]
    outmodelfile = args["outmodelfile"]
    ntrials = args["ntrials"]

    dfall = pd.read_csv(dataset)
    df = preprocess_dataset(dfall)

    X_train, X_test, y_train, y_test = split_train_test(df)
    # Train Model
    X, y = get_data(df)
    match algo:
        case "rf":
            best_params = random_forest_regression_optuna(X_train, y_train, ntrials)
            test_model(X_train, X_test, y_train, y_test, best_params, algo)
            model = RandomForestRegressor(**best_params, random_state=42)
        case "dt":
            best_params = dt_regression_optuna(X_train, y_train, ntrials)
            test_model(X_train, X_test, y_train, y_test, best_params, algo)
            model = DecisionTreeRegressor(**best_params, random_state=42)
        case _:
            # best_params = xgb_regression_optuna(X_train, y_train, ntrials)
            best_params = xgb_regression_gridsearch(X_train, y_train)
            test_model(X_train, X_test, y_train, y_test, best_params, algo)
            model = xgb.XGBRegressor(**best_params, random_state=42)

    model.fit(X, y)

    # Pickle and save file
    joblib.dump(model, outmodelfile)
    joblib.dump(category_mappings, "category_mappings.pkl")
    
    # predictallcombinations(model)


if __name__ == "__main__":
    ioflextrain()
