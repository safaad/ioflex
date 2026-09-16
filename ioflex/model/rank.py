import os
import warnings

warnings.filterwarnings("ignore")

import time
import argparse
import shlex
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool
from scipy.stats import qmc
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr
import joblib
from scipy.spatial.distance import cdist

from ioflex.common import (
    get_config_map,
    set_hints_with_ioflex,
    set_hints_env_romio,
    set_hints_env_cray,
    repair_cray_hints_valid_df,
    repair_cray_hints_valid,
    get_bandwidth_darshan,
    remove_path,
)

from ioflex.striping import setstriping

FEATURE_COLS = [
    "cb_nodes",
    "striping_factor",
    "striping_unit",
    "cray_cb_write_lock_mode",
    "romio_cb_read",
    "romio_cb_write",
    "romio_ds_read",
    "romio_ds_write",
    "romio_no_indep_rw",
]
SCALE_COLS = [
    "cb_nodes",
    "striping_factor",
    "striping_unit",
    "cray_cb_write_lock_mode",
    "romio_cb_read",
    "romio_cb_write",
    "romio_ds_read",
    "romio_ds_write",
]
HINT_ORDER = {"disable": 0, "automatic": 1, "enable": 2}
THREE_STATE = ["romio_cb_read", "romio_cb_write", "romio_ds_read", "romio_ds_write"]
LABEL_TO_GAIN = {0: 0, 1: 1, 2: 5, 3: 15}

LGBM_PARAMS = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "label_gain": [0, 1, 5, 15],
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 5,
    "n_estimators": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}
XGBM_LAMBDAMART_PARAMS = {
    "objective": "rank:pairwise",
    "ndcg_exp_gain": False,
    "eval_metric": ["ndcg@5", "ndcg@10"],
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "n_estimators": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
    "tree_method": "hist",
}
XGBM_LAMBDALOSS_PARAMS = {
    **XGBM_LAMBDAMART_PARAMS,
    "objective": "rank:ndcg",
    "lambdanorm": True,
}
CATBOOST_PARAMS = {
    "loss_function": "YetiRank",
    "eval_metric": "NDCG:top=10",
    "iterations": 50,
    "learning_rate": 0.05,
    "depth": 3,
    "l2_leaf_reg": 10,
    "verbose": 0,
}
CATBOOST_PAIRWISE_PARAMS = {
    **CATBOOST_PARAMS,
    "loss_function": "YetiRankPairwise",
}

MODEL_REGISTRY = {
    "lgbm_lambdarank": {
        "lib": "lgbm",
        "params": {**LGBM_PARAMS, "objective": "lambdarank"},
    },
    "lgbm_xendcg": {"lib": "lgbm", "params": LGBM_PARAMS},
    "xgb_lambdamart": {"lib": "xgb", "params": XGBM_LAMBDAMART_PARAMS},
    "xgb_lambdaloss": {"lib": "xgb", "params": XGBM_LAMBDALOSS_PARAMS},
    "catboost_yetirank": {"lib": "catboost", "params": CATBOOST_PARAMS},
    "catboost_yetipairwise": {"lib": "catboost", "params": CATBOOST_PAIRWISE_PARAMS},
}


# Generate All Valid Combinations of this Configuration Space
def generate_parameter_grid(config_space, num_nodes, num_ranks):
    # copy to avoid mutating the caller's config_space (it's reused elsewhere)
    config_space = dict(config_space)
    config_space["nodes"] = [num_nodes]
    config_space["num_ranks"] = [num_ranks]

    grid = ParameterGrid(config_space)
    if hints == "cray":
        dfall = repair_cray_hints_valid_df(pd.DataFrame(grid))
    else:
        dfall = pd.DataFrame(grid)

    return dfall


# Feature Transformation and Label Generation
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["cb_nodes", "striping_factor"]:
        df[col] = np.log2(df[col].clip(lower=1))
    df["striping_unit"] = np.log2((df["striping_unit"] / 1024).clip(lower=1))
    df["cray_cb_write_lock_mode"] = df["cray_cb_write_lock_mode"].astype(int)
    for col in THREE_STATE:
        mapped = df[col].str.strip().str.lower().map(HINT_ORDER)
        if mapped.isna().any():
            raise ValueError(
                f"'{col}' unmapped values: {df[col][mapped.isna()].unique()}"
            )
        df[col] = mapped
    #    df["romio_no_indep_rw"] = df["romio_no_indep_rw"].map({True: 1, False: 0})
    #    if df["romio_no_indep_rw"].isna().any():
    #        raise ValueError("'romio_no_indep_rw' has unmapped values.")

    mapped_bool = (
        df["romio_no_indep_rw"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0})
    )
    if mapped_bool.isna().any():
        raise ValueError(
            f"'romio_no_indep_rw' unmapped values: "
            f"{df['romio_no_indep_rw'][mapped_bool.isna()].unique()}"
        )
    df["romio_no_indep_rw"] = mapped_bool

    return df


def make_labels(ratio: pd.Series) -> pd.Series:
    q95 = np.quantile(ratio, 0.95)
    bins = [
        (-np.inf, 0.98),
        (0.98, 1.10),
        (1.10, q95),
        (q95, np.inf),
    ]
    labels = np.full(len(ratio), -1, dtype=int)
    for lbl, (lo, hi) in enumerate(bins):
        labels[(ratio > lo) & (ratio <= hi)] = lbl
    if (labels == -1).any():
        raise ValueError(f"Unmapped samples: {ratio[labels == -1].values}")
    return pd.Series(labels, index=ratio.index)


def make_xgb_gains(y_label: np.ndarray) -> np.ndarray:
    return np.vectorize(LABEL_TO_GAIN.get)(y_label).astype(float)


# Model Creation, Fitting, and Prediction
def make_model(model_name: str):
    cfg = MODEL_REGISTRY[model_name]
    if cfg["lib"] == "lgbm":
        return lgb.LGBMRanker(**cfg["params"], eval_at=[5, 10])
    elif cfg["lib"] == "xgb":
        return xgb.XGBRanker(**cfg["params"])
    elif cfg["lib"] == "catboost":
        return CatBoostRanker(**cfg["params"])
    else:
        raise ValueError(f"Unknown lib: {cfg['lib']}")


def fit_model(model, model_name: str, X: pd.DataFrame, y: np.ndarray):
    cfg = MODEL_REGISTRY[model_name]
    if cfg["lib"] == "lgbm":
        model.fit(X, y, group=[len(y)])
    elif cfg["lib"] == "xgb":
        y_gain = make_xgb_gains(y)
        model.fit(X, y_gain, qid=np.zeros(len(y_gain), dtype=int))
    elif cfg["lib"] == "catboost":
        y_gain = make_xgb_gains(y).astype(float)
        pool = Pool(data=X, label=y_gain, group_id=np.zeros(len(y_gain), dtype=int))
        model.fit(pool)
    return model


def predict_model(model, model_name: str, X_pool: pd.DataFrame) -> np.ndarray:
    cfg = MODEL_REGISTRY[model_name]
    if cfg["lib"] in ("lgbm", "xgb"):
        return model.predict(X_pool)
    elif cfg["lib"] == "catboost":
        pool = Pool(data=X_pool, group_id=np.zeros(len(X_pool), dtype=int))
        return model.predict(pool)


# Evaluation Metrics for Learning-to-Rank
def ndcg_at_k(y_true, y_score, k, label_gain=None):
    if label_gain is None:
        label_gain = [0, 1, 5, 15]
    order = np.argsort(y_score)[::-1][:k]
    gains = np.array([label_gain[l] for l in y_true[order]], dtype=float)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    ideal = np.array(
        sorted([label_gain[l] for l in y_true], reverse=True)[:k], dtype=float
    )
    idcg = np.sum(ideal / discounts)
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(y_true, y_score, good_threshold=2):
    order = np.argsort(y_score)[::-1]
    for rank, label in enumerate(y_true[order], start=1):
        if label >= good_threshold:
            return 1.0 / rank
    return 0.0


def precision_at_k(y_true, y_score, k, good_threshold=2):
    order = np.argsort(y_score)[::-1][:k]
    return (y_true[order] >= good_threshold).mean()


def degradation_rate_at_k(y_true, y_score, k, degradation_label=0):
    order = np.argsort(y_score)[::-1][:k]
    return (y_true[order] <= degradation_label).mean()


def best_config_found_ratio(y_ratios, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    best_found = y_ratios[order].max()
    true_best = y_ratios.max()
    return best_found / true_best if true_best > 0 else 0.0


def regret_at_k(y_ratios, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    best_found = y_ratios[order].max()
    true_best = y_ratios.max()
    return true_best - best_found


def scientific_evaluation(
    y_true,
    y_ratios,
    y_score,
    k_values=[5, 10],
    good_threshold=2,
    degradation_label=0,
) -> pd.Series:
    results = {}
    for k in k_values:
        results[f"NDCG@{k}"] = ndcg_at_k(y_true, y_score, k=k)
        results[f"Precision@{k}"] = precision_at_k(
            y_true, y_score, k=k, good_threshold=good_threshold
        )
        results[f"DegradRate@{k}"] = degradation_rate_at_k(
            y_true, y_score, k=k, degradation_label=degradation_label
        )
        results[f"BCFR@{k}"] = best_config_found_ratio(y_ratios, y_score, k=k)
        results[f"Regret@{k}"] = regret_at_k(y_ratios, y_score, k=k)
    results["MRR"] = mean_reciprocal_rank(
        y_true, y_score, good_threshold=good_threshold
    )
    if len(y_true) > 1:
        tau, tau_p = kendalltau(y_true, y_score)
        rho, rho_p = spearmanr(y_true, y_score)
    else:
        tau = tau_p = rho = rho_p = np.nan
    results["Kendall_τ"] = tau
    results["Kendall_τ_p"] = tau_p
    results["Spearman_ρ"] = rho
    results["Spearman_ρ_p"] = rho_p
    return pd.Series(results)


def print_metrics(metrics: pd.Series, label: str = "", k_values: list = [5, 10]):
    print(f"\n── Metrics{' — ' + label if label else ''} ──")
    for k in k_values:
        print(
            f"  NDCG@{k}={metrics[f'NDCG@{k}']:.3f}  "
            f"Precision@{k}={metrics[f'Precision@{k}']:.3f}  "
            f"DegradRate@{k}={metrics[f'DegradRate@{k}']:.1%}  "
            f"BCFR@{k}={metrics[f'BCFR@{k}']:.3f}  "
            f"Regret@{k}={metrics[f'Regret@{k}']:.4f}"
        )
    print(
        f"  MRR={metrics['MRR']:.3f}  "
        f"τ={metrics['Kendall_τ']:.3f} (p={metrics['Kendall_τ_p']:.2e})  "
        f"ρ={metrics['Spearman_ρ']:.3f} (p={metrics['Spearman_ρ_p']:.2e})"
    )


def score_pool(
    model,
    model_name,
    scaler,
    unlabelled_df,
    transform_features,
    FEATURE_COLS,
    SCALE_COLS,
    predict_model,
):
    pool_t = transform_features(unlabelled_df)
    pool_t[SCALE_COLS] = scaler.transform(pool_t[SCALE_COLS])
    X_pool = pool_t[FEATURE_COLS]

    scores = predict_model(model, model_name, X_pool)

    scored = unlabelled_df.copy()
    scored["score"] = scores
    return scored


def select_top_k(scored_pool: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top_k_actual = min(top_k, len(scored_pool))
    picked = scored_pool.nlargest(top_k_actual, "score").copy()
    picked["pick_reason"] = "exploit" # No exploration
    return picked


def decayed_explore_frac(
    iteration: int,
    n_iterations: int,
    start_frac: float = 0.3,
    end_frac: float = 0.1,
) -> float:
    """
    Linearly decays explore_frac from start_frac (early iterations, when
    broad coverage matters most) to end_frac (later iterations, once
    diverse regions have been seeded and exploit can be trusted more).
    """
    if n_iterations <= 1:
        return start_frac
    progress = (iteration - 1) / (n_iterations - 1)  # 0.0 at iter 1, 1.0 at last iter
    return start_frac + (end_frac - start_frac) * progress


def compute_novelty_scores(
    candidates_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    scaler,
    transform_features,
    FEATURE_COLS,
    SCALE_COLS,
) -> np.ndarray:
    """
    Novelty = distance in (scaled) feature space to the nearest already-labelled
    point. Higher = more unlike anything the model has seen, i.e. a genuine
    unexplored region rather than a random pick that might just be near
    something already known to be bad.
    """
    cand_t = transform_features(candidates_df)
    cand_t[SCALE_COLS] = scaler.transform(cand_t[SCALE_COLS])
    X_cand = cand_t[FEATURE_COLS].values

    lab_t = transform_features(labelled_df)
    lab_t[SCALE_COLS] = scaler.transform(lab_t[SCALE_COLS])
    X_lab = lab_t[FEATURE_COLS].values

    dists = cdist(X_cand, X_lab).min(axis=1)
    # normalize to [0, 1] for interpretability / combining with other scores
    spread = dists.max() - dists.min()
    novelty = (dists - dists.min()) / spread if spread > 0 else np.zeros_like(dists)
    return novelty


def select_top_k_with_exploration(
    scored_pool: pd.DataFrame,
    labelled_df: pd.DataFrame,
    scaler,
    top_k: int,
    explore_frac: float = 0.3,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Splits each pick batch into exploit (top-scoring) and explore
    (novelty-ranked from the remaining pool — most unlike anything
    already labelled, not just uniformly random).
    """
    top_k_actual = min(top_k, len(scored_pool))
    n_explore = (
        max(1, int(round(top_k_actual * explore_frac))) if top_k_actual > 1 else 0
    )
    n_exploit = top_k_actual - n_explore

    exploit = (
        scored_pool.nlargest(n_exploit, "score")
        if n_exploit > 0
        else scored_pool.iloc[0:0]
    )
    remaining = scored_pool.drop(index=exploit.index)
    n_explore = min(n_explore, len(remaining))

    if n_explore > 0:
        novelty = compute_novelty_scores(
            remaining, labelled_df, scaler, transform_features, FEATURE_COLS, SCALE_COLS
        )
        remaining = remaining.copy()
        remaining["novelty"] = novelty
        explore = remaining.nlargest(n_explore, "novelty").drop(columns=["novelty"])
    else:
        explore = remaining.iloc[0:0]

    picked = pd.concat([exploit, explore])
    picked["pick_reason"] = ["exploit"] * len(exploit) + ["explore"] * len(explore)
    return picked


def select_mmr(
    scored_pool: pd.DataFrame,
    labelled_df: pd.DataFrame,
    scaler,
    top_k: int,
    lambda_: float = 0.7,
) -> pd.DataFrame:
    """
    Maximal Marginal Relevance selection.
    Each pick maximises: lambda * model_score - (1-lambda) * max_similarity_to_selected
    lambda_=1.0 pure exploitation (same as select_top_k)
    lambda_=0.5 balanced exploit / explore
    lambda_=0.0 pure diversity (furthest-first)
    """
    from sklearn.metrics.pairwise import rbf_kernel

    top_k_actual = min(top_k, len(scored_pool))
    scores = scored_pool["score"].values
    s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    pool_t = transform_features(scored_pool)
    pool_t[SCALE_COLS] = scaler.transform(pool_t[SCALE_COLS])
    X_pool = pool_t[FEATURE_COLS].values

    selected_pos = []  # positions in X_pool
    remaining_pos = list(range(len(scored_pool)))

    for _ in range(top_k_actual):
        if not selected_pos:
            idx = int(np.argmax(s_norm))
        else:
            X_sel = X_pool[selected_pos]
            mmr = [
                lambda_ * s_norm[i]
                - (1 - lambda_) * rbf_kernel(X_pool[[i]], X_sel, gamma=1.0).max()
                for i in remaining_pos
            ]
            idx = remaining_pos[int(np.argmax(mmr))]
        selected_pos.append(idx)
        remaining_pos.remove(idx)

    picked = scored_pool.iloc[selected_pos].copy()
    picked["pick_reason"] = "exploit"  # MMR has no hard split
    return picked


def compute_adaptive_lambda(
    labelled_df: pd.DataFrame,
    unlabelled_df: pd.DataFrame,
    scaler,
    new_best_prev: bool,
    base_lambda: float = 0.7,
) -> float:
    """
    Reactively adjusts lambda:
    - Increases exploration when label-3 configs are scarce in labelled set
      (proxy for pool exhaustion since pool labels are unknown in production)
    - Increases exploration when pool is highly novel (far from labelled)
    - Decreases exploration (exploits more) when a new best was just found
    """
    # label-3 fraction in labelled set as proxy for how much good region remains
    lab_labels = make_labels(labelled_df["ratio"])
    label3_frac = (lab_labels == 3).mean()

    # average novelty of pool vs labelled in scaled feature space
    lab_t = transform_features(labelled_df)
    lab_t[SCALE_COLS] = scaler.transform(lab_t[SCALE_COLS])
    pool_t = transform_features(unlabelled_df)
    pool_t[SCALE_COLS] = scaler.transform(pool_t[SCALE_COLS])

    dists = cdist(pool_t[FEATURE_COLS].values, lab_t[FEATURE_COLS].values).min(axis=1)
    spread = dists.max() - dists.min()
    novelty = (dists - dists.min()) / spread if spread > 0 else np.zeros_like(dists)
    avg_novelty = novelty.mean()

    lam = float(
        np.clip(
            base_lambda
            - (1 - label3_frac) * 0.3  # scarce good configs explore more
            + avg_novelty
            * 0.2,  # to be revisited, lean on model score avoid blind exploration
            0.1,
            1.0,
        )
    )

    # React to last iteration's outcome
    if new_best_prev:
        lam = max(0.10, lam - 0.05)  # found new best exploit neighbourhood
    else:
        lam = min(0.60, lam + 0.05)  # stuck push exploration

    print(
        f"  adaptive lambda: {lam:.3f}  "
        f"(label3_frac={label3_frac:.2f}, pool_novelty={avg_novelty:.2f}, "
        f"new_best_prev={new_best_prev})"
    )
    return lam


def make_labels_with_boost(
    ratio: pd.Series,
    X_labelled: pd.DataFrame,  
    X_best: np.ndarray,
    boost_radius: float = 0.3,
) -> pd.Series:
    """
    Promotes configs similar to the best known config one label tier up.
    Steers the model to focus on the promising neighbourhood after a new best.
    boost_radius: RBF similarity threshold (0–1). Lower = wider neighbourhood.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    base_labels = make_labels(ratio)
    sim = rbf_kernel(X_labelled.values, X_best.reshape(1, -1), gamma=1.0).flatten()
    near = sim >= boost_radius
    boosted = base_labels.copy()
    boosted[near] = np.minimum(boosted[near] + 1, 3)
    n_boosted = int(near.sum())
    if n_boosted:
        print(
            f"  label boost: promoted {n_boosted} configs "
            f"(similarity ≥ {boost_radius})"
        )
    return boosted


def apply_fallback(
    scored_pool: pd.DataFrame,
    labelled_df: pd.DataFrame,
    scaler,
    top_k: int,
    lambda_: float,
    min_label3: int = 2,
    random_state: int = None,
) -> tuple[pd.DataFrame, str]:
    """
    When label-3 configs are scarce in the labelled set (proxy for pool
    exhaustion), fall back to high-diversity MMR with a low lambda rather
    than pure exploitation — avoids wasting all picks on the same depleted
    region.
    Returns (picked_candidates, mode_used).
    """
    lab_labels = make_labels(labelled_df["ratio"])
    n_label3 = (lab_labels == 3).sum()

    if n_label3 < min_label3:
        # Force exploration: override lambda to a low value
        eff_lambda = min(0.2, lambda_)
        mode = f"fallback-diversity (label3={n_label3}<{min_label3})"
        print(f"  {mode} lambda overridden to {eff_lambda:.3f}")
    else:
        eff_lambda = lambda_
        mode = "normal"

    picked = select_mmr(
        scored_pool,
        labelled_df,
        scaler,
        top_k=top_k,
        lambda_=eff_lambda,
    )
    return picked, mode


def active_learning_loop_single(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    default_value: float,
    model_name: str = "xgb_lambdaloss",
    n_iterations: int = 5,
    top_k: int = 10,
    k_values: list = [5, 10],
    explore_frac_start: float = 0.3,
    explore_frac_end: float = 0.1,
    random_state: int = None,
    selection_mode: str = "mmr",  # "mmr" | "split" | "default"
    base_lambda: float = 0.7,  # starting lambda for MMR / adaptive
    label_boost: bool = True,  # enable soft label gain shift
    boost_radius: float = 0.3,  # RBF similarity threshold for boost
    fallback: bool = True,  # enable fallback when label-3 scarce
    min_label3: int = 2,  # threshold for fallback trigger
):
    assert (
        model_name in MODEL_REGISTRY
    ), f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
    assert selection_mode in ("mmr", "split", "default"), "selection_mode must be 'mmr', 'split', or 'default'"

    print(
        f"\n  active_learning config: selection={selection_mode}  "
        f"base_lambda={base_lambda}  label_boost={label_boost}"
        f"(r={boost_radius})  fallback={fallback}(min_label3={min_label3})"
    )

    labelled = df_train.copy()
    unlabelled = df_test.copy()
    df_picked = pd.DataFrame()
    history = []
    model = None
    scaler = None
    best_X = None  # scaled feature vector of best config found so far
    current_lam = base_lambda
    current_explore_frac = explore_frac_start
    new_best_prev = False  # outcome of the previous iteration

    for iteration in range(1, n_iterations + 1):

        print(
            f"\nIteration {iteration} | model={model_name} | "
            f"labelled={len(labelled)} unlabelled={len(unlabelled)} | "
        )

        if len(unlabelled) == 0:
            print("Pool exhausted — stopping.")
            break

        q95 = np.quantile(labelled["ratio"], 0.95)
        best = labelled["ratio"].max()
        threshold = q95

        df_t = transform_features(labelled)

        scaler = StandardScaler()
        df_t[SCALE_COLS] = scaler.fit_transform(df_t[SCALE_COLS])
        
        
        if label_boost and best_X is not None:
            df_t["label"] = make_labels_with_boost(
                labelled["ratio"],
                X_labelled=df_t[FEATURE_COLS],
                X_best=best_X,
                boost_radius=boost_radius,
            )
        else:
            df_t["label"] = make_labels(labelled["ratio"])

        print(f"\nLabel distribution (iteration {iteration})")
        print(
            df_t["label"]
            .value_counts()
            .sort_index()
            .rename(
                {
                    0: "0: degradation",
                    1: "1: neutral",
                    2: "2: mild imp.",
                    3: "3: significant imp.",
                }
            )
            .to_string()
        )

        X = df_t[FEATURE_COLS]
        y = df_t["label"].values


        model = make_model(model_name)
        fit_model(model, model_name, X, y)

        scored_pool = score_pool(
            model,
            model_name,
            scaler,
            unlabelled,
            transform_features,
            FEATURE_COLS,
            SCALE_COLS,
            predict_model,
        )

        if selection_mode == "mmr":

            current_lam = compute_adaptive_lambda(
                labelled_df=labelled,
                unlabelled_df=unlabelled,
                scaler=scaler,
                new_best_prev=new_best_prev,
                base_lambda=base_lambda,
            )
            
            print(f"selection={selection_mode} | adaptive_lambda={current_lam:.3f}")
            # MMR with fallback
            if fallback:
                picked_candidates, sel_mode = apply_fallback(
                    scored_pool=scored_pool,
                    labelled_df=labelled,
                    scaler=scaler,
                    top_k=top_k,
                    lambda_=current_lam,
                    min_label3=min_label3,
                    random_state=random_state,
                )
            else:
                picked_candidates = select_mmr(
                    scored_pool=scored_pool,
                    labelled_df=labelled,
                    scaler=scaler,
                    top_k=top_k,
                    lambda_=current_lam,
                )
                sel_mode = "mmr"
        elif selection_mode == "split":
            
            current_explore_frac = decayed_explore_frac(
                iteration, n_iterations, explore_frac_start, explore_frac_end
            )
            # Original split strategy kept as-is
            picked_candidates = select_top_k_with_exploration(
                scored_pool,
                labelled_df=labelled,
                scaler=scaler,
                top_k=top_k,
                explore_frac=current_explore_frac,
                random_state=random_state,
            )
            sel_mode = "split"
        else:
            # Default Select top-k
            picked_candidates = select_top_k(scored_pool, top_k=top_k)
            sel_mode = "default"
            

        print(
            f"  picks ({sel_mode}): "
            f"{(picked_candidates['pick_reason'] == 'exploit').sum()} exploit, "
            f"{(picked_candidates['pick_reason'] == 'explore').sum()} explore"
        )

        picked = eval_runs(
            picked_candidates.drop(columns=["score", "pick_reason"]),
            default_value,
        )
        if picked.empty:
            print("All picks failed labeling this iteration — skipping update.")
            continue

        # Realign scores and pick_reason to successfully labelled rows
        picked = picked.copy()
        picked["score"] = picked_candidates.loc[picked.index, "score"]
        picked["pick_reason"] = picked_candidates.loc[picked.index, "pick_reason"]

        mean_ratio = picked["ratio"].mean()
        best_ratio = picked["ratio"].max()
        new_best = best_ratio > best
        picked_labels = make_labels(picked["ratio"])

        print(
            f"Iteration {iteration} results: mean_ratio={mean_ratio:.4f} "
            f"best_ratio={best_ratio:.4f} new_best={new_best}"
        )

        batch_metrics = scientific_evaluation(
            y_true=picked_labels.values,
            y_ratios=picked["ratio"].values,
            y_score=picked["score"].values,
            k_values=[k for k in k_values if k <= len(picked)],
        )
        print_metrics(
            batch_metrics,
            label=f"iter {iteration}",
            k_values=[k for k in k_values if k <= len(picked)],
        )

        # Track explore contribution
        explore_mask = picked["pick_reason"] == "explore"
        explore_best_ratio = (
            picked.loc[explore_mask, "ratio"].max() if explore_mask.any() else np.nan
        )
        explore_found_new_best = bool(
            explore_mask.any() and picked.loc[explore_mask, "ratio"].max() > best
        )

        if new_best:
            best_idx = picked["ratio"].idxmax()
            # Store raw-transform feature vector (before scaling)
            best_row = transform_features(
                picked.loc[[best_idx]].drop(
                    columns=["score", "pick_reason", "ratio"], errors="ignore"
                )
            )
            # Scale with current scaler so boost uses same space as labels
            best_row_scaled = best_row[FEATURE_COLS].copy()
            best_row_scaled[SCALE_COLS] = scaler.transform(best_row_scaled[SCALE_COLS])
            best_X = best_row_scaled[FEATURE_COLS].values[0]

        new_best_prev = new_best

        iter_row = {
            "iteration": iteration,
            "active_model": model_name,
            "n_labelled": len(labelled),
            "q95_threshold": threshold,
            "best_so_far": best,
            "mean_ratio": mean_ratio,
            "best_ratio": best_ratio,
            "new_best": new_best,
            "selection_mode": sel_mode,
            "lambda": current_lam,
            "degradation_rate": (picked_labels == 0).mean(),
            "precision": (picked_labels >= 2).mean(),
            "explore_frac": current_explore_frac,
            "n_exploit": int((~explore_mask).sum()),
            "n_explore": int(explore_mask.sum()),
            "explore_best_ratio": explore_best_ratio,
            "explore_found_new_best": explore_found_new_best,
            **batch_metrics.to_dict(),
        }
        history.append(iter_row)

        picked_for_merge = picked.drop(columns=["score", "pick_reason"])
        df_picked = pd.concat([df_picked, picked_for_merge], ignore_index=True)
        labelled = pd.concat([labelled, picked_for_merge], ignore_index=True)
        unlabelled = unlabelled.drop(index=picked_candidates.index).reset_index(
            drop=True
        )

    df_history = pd.DataFrame(history)
    return labelled, model, scaler, df_history, df_picked


def generate_lhs_samples(config_space, nsamples):

    param_keys = list(config_space.keys())
    param_grid = list(config_space.values())
    n_dims = len(param_grid)
    lhs = qmc.LatinHypercube(d=n_dims, optimization="lloyd")

    print(f"Number of samples is: {nsamples}")
    lhs_samples = lhs.random(n=nsamples)

    sampled_configs = []
    for i in range(nsamples):
        sampled_config = []
        for j, param_values in enumerate(param_grid):
            index = int(np.floor(lhs_samples[i, j] * len(param_values)))
            index = min(index, len(param_values) - 1)  # guard against 1.0 edge case
            sampled_config.append(param_values[index])
        sample_instance = dict(zip(param_keys, sampled_config))
        if hints == "cray":
            repair_cray_hints_valid(sample_instance, num_ranks, num_nodes)
        sampled_configs.append(sample_instance)

    return sampled_configs


def compute_ratio(tune_bandwidth: bool, measured: float, default_value: float):
    if measured is None or not default_value:
        return None
    if tune_bandwidth:
        return measured / default_value
    else:
        return default_value / measured


def eval_runs(
    picked_df: pd.DataFrame,
    default_value: float,
) -> pd.DataFrame:

    picked_df = picked_df.copy()
    ratios = []
    elapsedtimes = []
    bandwidths = []

    dir_path = os.environ.get("PWD", os.getcwd())
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    for idx, row in picked_df.iterrows():
        sample_instance = row.to_dict()

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

        stripe_count = int(sample_instance.get("striping_factor", 8))
        stripe_size = (
            str(sample_instance["striping_unit"] // 1048576) + "M"
            if "striping_unit" in sample_instance
            else "1M"
        )
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
        elapsed = time.time() - start_time

        sample_instance["elapsedtime"] = elapsed
        elapsedtimes.append(elapsed)
        if tune_bandwidth:
            darshan_dir = os.environ["DARSHAN_LOG_DIR_PATH"]
            log_path = os.path.join(darshan_dir, "*.darshan")
            objective = get_bandwidth_darshan(log_path, "MPI-IO")
            if objective == -1:
                print(
                    "Darshan file wasn't properly generated. "
                    "Check the Darshan settings or the application correctness"
                )
                bandwidths.append(None)
                ratios.append(None)
                for f in files_to_clean:
                    remove_path(f)
                continue
            sample_instance["I/O-Bandwidth-Mib/s"] = objective
            measured = objective
            bandwidths.append(objective)
        else:
            measured = elapsed

        ratio = compute_ratio(tune_bandwidth, measured, default_value)
        ratios.append(ratio)

        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"Labeled config idx={idx} ratio={ratio_str} (tune_bandwidth={tune_bandwidth})")

        configs_str = ", ".join(f"{k}: {v}" for k, v in sample_instance.items())
        if logisset:
            logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
            logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")
        print(f"Running config: {configs_str}")

        for f in files_to_clean:
            remove_path(f)

    picked_df["ratio"] = ratios
    picked_df["elapsedtime"] = elapsedtimes
    if tune_bandwidth:
        picked_df["I/O-Bandwidth-Mib/s"] = bandwidths
    failed = picked_df["ratio"].isna().sum()
    if failed:
        print(f"{failed} picks failed labeling and are dropped.")
    picked_df = picked_df.dropna(
        subset=["ratio"]
    )  # keep original index
    return picked_df


def run(args=None):

    ap = argparse.ArgumentParser(prog="ioflex model --rank")
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./TopRankedConfigs.csv",
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
        "--default_value",
        type=float,
        required=True,
        help="Baseline value matching --tune_bandwidth "
        "(default bandwidth if set, default elapsed time otherwise) ",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent.parent / "configs" / "tune_config_romio.json",
        help="Path to JSON configuration file (default: ../configs/tune_config_romio.json",
    )
    ap.add_argument(
        "--nsamples", type=int, default=30, help="Number of pool LHS samples"
    )
    ap.add_argument(
        "--samples_csv", type=str, default=None, help="CSV of pre-generated configs"
    )
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

    ap.add_argument(
        "--model_name",
        type=str,
        default="xgb_lambdaloss",
        help="Learning-to-rank model name",
        choices=[
            "lgbm_lambdarank",
            "lgbm_xendcg",
            "xgb_lambdamart",
            "xgb_lambdaloss",
            "catboost_yetirank",
            "catboost_yetipairwise",
        ],
    )
    ap.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of active learning iterations",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of candidates picked and labeled per iteration",
    )
    ap.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=[5, 10],
        help="k values for evaluation metrics",
    )

    ap.add_argument(
        "--selection_mode",
        type=str,
        default="mmr",
        choices=["mmr", "split", "default"],
        help=(
            "Candidate selection strategy. "
            "'mmr' = Maximal Marginal Relevance (diverse + high-score). "
            "'split' = hard exploit/explore split with novelty-ranked explore."
            "'default' = select top-k scored candidates"
        ),
    )
    ap.add_argument(
        "--base_lambda",
        type=float,
        default=0.7,
        help=(
            "Base lambda for MMR / adaptive lambda. "
            "1.0 = pure exploitation, 0.0 = pure diversity."
        ),
    )
    ap.add_argument(
        "--label_boost",
        action="store_true",
        default=True,
        help="Enable soft label gain shift after a new best config is found.",
    )
    ap.add_argument(
        "--no_label_boost",
        dest="label_boost",
        action="store_false",
        help="Disable label boost.",
    )
    ap.add_argument(
        "--boost_radius",
        type=float,
        default=0.3,
        help=(
            "RBF similarity threshold for label boost. "
            "Higher = tighter neighbourhood around best config."
        ),
    )
    ap.add_argument(
        "--fallback",
        action="store_true",
        default=True,
        help=(
            "Enable fallback to high-diversity MMR when label-3 configs "
            "are scarce in the labelled set."
        ),
    )
    ap.add_argument(
        "--no_fallback",
        dest="fallback",
        action="store_false",
        help="Disable fallback.",
    )
    ap.add_argument(
        "--min_label3",
        type=int,
        default=2,
        help="Minimum label-3 configs in labelled set before fallback triggers.",
    )
    ap.add_argument(
        "--explore_frac_start",
        type=float,
        default=0.3,
        help="Starting explore fraction for split mode / decayed schedule.",
    )
    ap.add_argument(
        "--explore_frac_end",
        type=float,
        default=0.1,
        help="Ending explore fraction for split mode / decayed schedule.",
    )
    args = vars(ap.parse_args(args))

    global num_ranks, num_nodes, ioflexset, run_app, logisset, logfile_o, logfile_e, hints, tune_bandwidth, files_to_clean, files_to_stripe
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    tune_bandwidth = args["tune_bandwidth"]
    default_value = args["default_value"]

    if not default_value:
        raise ValueError(f"--default_value must be a positive number, got {default_value!r}")
    
    outfilepath = args["outfile"]
    outdir = os.path.dirname(os.path.abspath(outfilepath))
    os.makedirs(outdir, exist_ok=True)

    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]
    nsamples = args["nsamples"]
    samples_csv = args["samples_csv"]

    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    hints = args["with_hints"]
    config_path = args["config"]

    CONFIG_MAP, files_to_clean, files_to_stripe = get_config_map(hints, config_path)
    config_space = {key: value for key, value in sorted(CONFIG_MAP.items()) if value}

    if samples_csv:
        print(f"Loading configs from {samples_csv}")
        df_train = pd.read_csv(samples_csv)
        if df_train.empty:
            raise ValueError(f"--samples_csv {samples_csv} is empty")
    else:
        lhs_samples = generate_lhs_samples(config_space, nsamples)
        df_train = pd.DataFrame(lhs_samples)
        df_train["nodes"] = num_nodes
        df_train["num_ranks"] = num_ranks

    # Label the seed set if it isn't labelled yet
    if "ratio" not in df_train.columns:
        if samples_csv is None:
            samples_csv = os.path.join(outdir, "labelled_samples.csv")
        df_train = eval_runs(df_train, default_value)
        df_train.to_csv(samples_csv, index=False)

    df_test = generate_parameter_grid(config_space, num_nodes, num_ranks)

    shared_cols = [c for c in df_test.columns if c in df_train.columns]
    mask = ~df_test.set_index(shared_cols).index.isin(
        df_train.set_index(shared_cols).index
    )
    df_test = df_test[mask].reset_index(drop=True)

    labelled, model, scaler, df_history, df_picked = active_learning_loop_single(
        df_train=df_train,
        df_test=df_test,
        default_value=default_value,
        model_name=args["model_name"],
        n_iterations=args["n_iterations"],
        top_k=args["top_k"],
        k_values=args["k_values"],
        explore_frac_start=args.get("explore_frac_start", 0.3),
        explore_frac_end=args.get("explore_frac_end", 0.1),
        random_state=args.get("random_state"),
        selection_mode=args.get("selection_mode", "mmr"),
        base_lambda=args.get("base_lambda", 0.7),
        label_boost=args.get("label_boost", True),
        boost_radius=args.get("boost_radius", 0.3),
        fallback=args.get("fallback", True),
        min_label3=args.get("min_label3", 2),
    )

    history_path = os.path.join(outdir, "active_learning_history.csv")
    df_history.to_csv(history_path, index=False)
    df_picked.to_csv(outfilepath, index=False)
    print(f"Saved history to {history_path}")
    print(f"Saved picked configs to {outfilepath}")

    if model is not None:
        model_path = os.path.join(outdir, "model.pkl")
        joblib.dump({"model": model, "scaler": scaler}, model_path)
        print(f"Saved trained model to {model_path}")

    if logisset:
        logfile_o.close()
        logfile_e.close()
