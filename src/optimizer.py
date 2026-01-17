import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from utils import setup_logger


def _build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    categorical_pipe = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe.append(("onehot", encoder))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_pipe), numeric_cols),
            ("cat", Pipeline(categorical_pipe), categorical_cols),
        ],
        remainder="drop",
    )


def _coerce_numeric(df: pd.DataFrame, numeric_cols):
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def load_data(data_dir: Path, index_col: str, logger: logging.Logger | None = None):
    X_train = pd.read_excel(data_dir / "train.xlsx")
    y_train_df = pd.read_excel(data_dir / "train_target.xlsx")

    if index_col in X_train.columns:
        X_train = X_train.drop(columns=[index_col])

    y_train = y_train_df.iloc[:, 0].astype(float).values
    target_name = y_train_df.columns[0]

    if logger:
        logger.info("Loaded data: train=%s, target=%s", X_train.shape, target_name)

    return X_train, y_train, target_name


def evaluate_cv(X, y, numeric_cols, categorical_cols, seed, folds, params):
    y_log = np.log1p(y)
    y_bins = pd.qcut(y, q=folds, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_bins):
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr_log, y_val_log = y_log[tr_idx], y_log[val_idx]
        y_val_orig = y[val_idx]

        _coerce_numeric(X_tr, numeric_cols)
        _coerce_numeric(X_val, numeric_cols)

        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr_t,
            y_tr_log,
            eval_set=[(X_val_t, y_val_log)],
            use_best_model=True,
            verbose=False,
        )

        pred = np.expm1(model.predict(X_val_t))
        scores.append(r2_score(y_val_orig, pred))

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--index-col", type=str, default="TaskID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(Path(args.log_dir), log_name="optimizer.log")

    X, y, target_name = load_data(data_dir, index_col=args.index_col, logger=logger)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    def objective(trial: optuna.Trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "od_type": "Iter",
            "od_wait": trial.suggest_int("od_wait", 30, 150, step=10),
            "random_seed": args.seed,
            "task_type": "CPU",
        }
        score = evaluate_cv(
            X, y, numeric_cols, categorical_cols, args.seed, args.folds, params
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, n_jobs=-1)

    best_params = study.best_params
    best_score = float(study.best_value)

    logger.info("Best val_mean R2: %.6f", best_score)
    logger.info("Best params: %s", best_params)

    params_path = out_dir / "best_params.json"
    results_path = out_dir / "tuner_results.json"

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_val_r2": best_score, "best_params": best_params},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Best val_mean R2: {best_score:.6f}")
    print(f"Saved params to {params_path}")


if __name__ == "__main__":
    main()
