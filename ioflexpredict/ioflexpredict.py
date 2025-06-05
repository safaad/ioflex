import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
import xgboost as xgb
import argparse
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import itertools
import joblib
import os, sys
import sys, os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ioflexheader import get_config_map, build_category_map, smape
from utils import header


def random_forest_regression_optuna(X, y, n_trials=100):

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
            "random_state": 42,
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
            "objective": "reg:squarederror",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "random_state": 42,
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
        "colsample_bytree": [0.8, 1.0],
    }

    # Initialize XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Perform GridSearchCV for hyperparameter tuning
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=kf,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    return best_params


def preprocess_dataset(df):
    dfnew = pd.DataFrame()
    for col in CONFIG_MAP.keys():
        dfnew[col] = df[col].map(CATEGORY_MAP[col]).fillna(-1).astype(int)

    if "striping_factor" in CONFIG_MAP:
        dfnew["striping_factor"] = df["striping_factor"].astype(int)
    if "cb_nodes" in CONFIG_MAP:
        dfnew["cb_nodes"] = df["cb_nodes"].astype(int)

    dfnew["runtime"] = df["runtime"]
    X = dfnew.loc[:, (dfnew.columns != "runtime")].to_numpy()
    y = dfnew["runtime"].to_numpy()
    return X, y


def split_train_test(df):
    # make sure to remove overlapping values
    df["config"] = df.loc[:, df.columns != "runtime"].astype(str).agg("-".join, axis=1)
    train_conf, test_conf = train_test_split(
        df["config"].unique(), random_state=123, test_size=0.25, shuffle=True
    )
    train = df.loc[df["config"].isin(train_conf)]
    test = df.loc[df["config"].isin(test_conf)]

    train = train.drop("config", axis=1)
    test = test.drop("config", axis=1)

    X_train, y_train = preprocess_dataset(train)
    X_test, y_test = preprocess_dataset(test)

    return X_train, X_test, y_train, y_test


def test_model(df, best_params, algo):

    X_train, X_test, y_train, y_test = split_train_test(df)
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


def predictallcombinations(model):

    combinations = list(itertools.product(CONFIG_MAP.items()))

    # Create a Pandas DataFrame
    dfall = pd.DataFrame(
        combinations,
        columns=[CONFIG_MAP.keys()],
    )
    df = pd.DataFrame()
    for col in CONFIG_MAP.keys():
        df[col] = dfall[col].map(CATEGORY_MAP[col]).fillna(-1).astype(int)

    if "striping_factor" in CONFIG_MAP:
        df["striping_factor"] = dfall["striping_factor"].astype(int)
    if "cb_nodes" in CONFIG_MAP:
        df["cb_nodes"] = dfall["cb_nodes"].astype(int)

    X = df.loc[:, :].to_numpy()

    y_hat = model.predict(X)

    df["pred-time"] = y_hat
    df.to_csv("predicted.csv")


def predict_instance(model, instance):

    for col, mapping in CATEGORY_MAP.items():
        instance[col] = mapping.get(instance[col], -1)

    instance_df = pd.DataFrame([instance])
    prediction = model.predict(instance_df)[0]
    return prediction


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
    ap.add_argument(
        "--with_hints",
        type=str,
        default="romio",
        help="MPIIO hints mode",
        choices=["romio", "cray", "ompio"],
    )

    parse = ap.parse_args()
    args = vars(parse)

    algo = args["algo"]
    dataset = args["dataset"]
    outmodelfile = args["outmodelfile"]
    ntrials = args["ntrials"]
    global CONFIG_MAP, CATEGORY_MAP

    # Define configurations mappings
    hints = args["with_hints"]

    CONFIG_MAP = get_config_map(hints)
    CATEGORY_MAP = build_category_map(CONFIG_MAP)

    df = pd.read_csv(dataset)

    # Train Model
    X, y = preprocess_dataset(df)
    match algo:
        case "rf":
            best_params = random_forest_regression_optuna(X, y, ntrials)
            test_model(df, best_params, algo)
            model = RandomForestRegressor(**best_params, random_state=42)
        case "dt":
            best_params = dt_regression_optuna(X, y, ntrials)
            test_model(df, best_params, algo)
            model = DecisionTreeRegressor(**best_params, random_state=42)
        case _:
            # best_params = xgb_regression_optuna(X, y, ntrials)
            best_params = xgb_regression_gridsearch(X, y)
            test_model(df, best_params, algo)
            model = xgb.XGBRegressor(**best_params, random_state=42)

    model.fit(X, y)

    # Pickle and save file
    joblib.dump(model, outmodelfile)

    # predictallcombinations(model)


if __name__ == "__main__":
    header.printheader()
    ioflextrain()
