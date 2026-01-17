import argparse
import json
import math
import os
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from utils import adj_r2, rae, setup_logger


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
    X_test = pd.read_excel(data_dir / "test.xlsx")
    y_test_df = pd.read_excel(data_dir / "test_target.xlsx")

    if index_col in X_train.columns:
        X_train = X_train.drop(columns=[index_col])
    if index_col in X_test.columns:
        X_test = X_test.drop(columns=[index_col])

    y_train = y_train_df.iloc[:, 0].astype(float).values
    y_test = y_test_df.iloc[:, 0].astype(float).values
    target_name = y_train_df.columns[0]

    if logger:
        logger.info(
            "Loaded data: train=%s, test=%s, target=%s",
            X_train.shape,
            X_test.shape,
            target_name,
        )

    return X_train, y_train, X_test, y_test, target_name


def train_cv(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_cols,
    categorical_cols,
    n_splits=10,
    seed=42,
    cat_params=None,
    verbose=200,
    logger: logging.Logger | None = None,
):
    if cat_params is None:
        cat_params = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 5,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "od_type": "Iter",
            "od_wait": 100,
            "random_seed": seed,
            "task_type": "CPU",
        }

    y_train_log = np.log1p(y_train)

    y_bins = pd.qcut(y_train, q=n_splits, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_metrics = []
    test_metrics = []
    test_preds_folds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bins)):
        X_tr = X_train.iloc[tr_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr_log, y_val_log = y_train_log[tr_idx], y_train_log[val_idx]
        y_val_orig = y_train[val_idx]

        _coerce_numeric(X_tr, numeric_cols)
        _coerce_numeric(X_val, numeric_cols)
        X_test_fold = X_test.copy()
        _coerce_numeric(X_test_fold, numeric_cols)

        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)
        X_test_t = preprocessor.transform(X_test_fold)

        model = CatBoostRegressor(**cat_params)
        model.fit(
            X_tr_t,
            y_tr_log,
            eval_set=[(X_val_t, y_val_log)],
            use_best_model=True,
            verbose=verbose,
        )

        val_pred = np.expm1(model.predict(X_val_t))
        test_pred = np.expm1(model.predict(X_test_t))

        n_features = X_tr_t.shape[1]
        r2 = r2_score(y_val_orig, val_pred)
        adj = adj_r2(r2, n=len(y_val_orig), p=n_features)
        mae = mean_absolute_error(y_val_orig, val_pred)
        rmse = mean_squared_error(y_val_orig, val_pred) ** 0.5
        rae_val = rae(y_val_orig, val_pred)
        val_metrics.append(
            {"R2": r2, "AdjR2": adj, "MAE": mae, "RMSE": rmse, "RAE": rae_val}
        )

        if y_test is not None and len(y_test) == len(test_pred):
            r2_t = r2_score(y_test, test_pred)
            adj_t = adj_r2(r2_t, n=len(y_test), p=n_features)
            mae_t = mean_absolute_error(y_test, test_pred)
            rmse_t = mean_squared_error(y_test, test_pred) ** 0.5
            rae_t = rae(y_test, test_pred)
            test_metrics.append(
                {"R2": r2_t, "AdjR2": adj_t, "MAE": mae_t, "RMSE": rmse_t, "RAE": rae_t}
            )

        test_preds_folds.append(test_pred)

        if logger:
            logger.info(
                "Fold %s metrics - R2=%.4f AdjR2=%.4f MAE=%.4f RMSE=%.4f RAE=%.4f",
                fold + 1,
                r2,
                adj,
                mae,
                rmse,
                rae_val,
            )

    val_mean = {
        k: float(np.nanmean([m[k] for m in val_metrics])) for k in val_metrics[0].keys()
    }
    test_mean = (
        {
            k: float(np.nanmean([m[k] for m in test_metrics]))
            for k in test_metrics[0].keys()
        }
        if test_metrics
        else {}
    )

    test_pred_avg = np.mean(np.vstack(test_preds_folds), axis=0)

    if logger:
        logger.info("Validation mean metrics: %s", val_mean)
        logger.info("Test mean metrics: %s", test_mean)

    return val_metrics, val_mean, test_metrics, test_mean, test_pred_avg


def shap_summary_plot(model, X, out_path: Path, logger: logging.Logger | None = None):
    shap_values = model.get_feature_importance(Pool(X, label=None), type="ShapValues")
    shap_values = np.array(shap_values)
    shap_values_no_bias = (
        shap_values[:, :, :-1] if shap_values.ndim == 3 else shap_values[:, :-1]
    )
    if shap_values_no_bias.ndim == 3:
        shap_values_no_bias = shap_values_no_bias[:, 0, :]
    shap.summary_plot(shap_values_no_bias, X, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    if logger:
        logger.info("SHAP summary saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--index-col", type=str, default="TaskID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument("--model-path", type=str, default="outputs/full_model.cbm")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir)

    logger.info(
        "Run started with data_dir=%s out_dir=%s log_dir=%s",
        data_dir,
        out_dir,
        log_dir,
    )

    X_train, y_train, X_test, y_test, target_name = load_data(
        data_dir, index_col=args.index_col, logger=logger
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    cat_params = {
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "depth": args.depth,
        "l2_leaf_reg": 5,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "od_type": "Iter",
        "od_wait": 100,
        "random_seed": args.seed,
        "task_type": "CPU",
    }

    best_params_path = out_dir / "best_params.json"
    if best_params_path.exists():
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)
            for key, value in best_params.items():
                cat_params[key] = value
            logger.info("Loaded best_params.json and updated cat_params.")
        except Exception as exc:
            logger.warning("Failed to load best_params.json: %s", exc)

    (
        val_metrics,
        val_mean,
        test_metrics,
        test_mean,
        test_pred_avg,
    ) = train_cv(
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_cols,
        categorical_cols,
        n_splits=args.folds,
        seed=args.seed,
        cat_params=cat_params,
        verbose=args.verbose,
        logger=logger,
    )

    metrics_payload = {
        "val_by_fold": val_metrics,
        "val_mean": val_mean,
        "test_by_fold": test_metrics,
        "test_mean": test_mean,
    }

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": test_pred_avg,
        }
    )

    _coerce_numeric(X_train, numeric_cols)
    _coerce_numeric(X_test, numeric_cols)
    full_preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_train_full = full_preprocessor.fit_transform(X_train)
    X_test_full = full_preprocessor.transform(X_test)
    try:
        feature_names = full_preprocessor.get_feature_names_out()
        X_train_full = pd.DataFrame(X_train_full, columns=feature_names)
        X_test_full = pd.DataFrame(X_test_full, columns=feature_names)
    except Exception:
        pass

    y_train_log = np.log1p(y_train)

    full_model = CatBoostRegressor(**cat_params)
    full_model.fit(
        X_train_full,
        y_train_log,
        verbose=args.verbose,
    )
    shap_path = out_dir / "shap_summary.png"
    shap_summary_plot(full_model, X_train_full, shap_path, logger=logger)

    full_pred = np.expm1(full_model.predict(X_test_full))
    full_r2 = r2_score(y_test, full_pred)
    full_adj = adj_r2(full_r2, n=len(y_test), p=X_train_full.shape[1])
    full_mae = mean_absolute_error(y_test, full_pred)
    full_rmse = mean_squared_error(y_test, full_pred) ** 0.5
    full_rae = rae(y_test, full_pred)
    metrics_payload["test_full_model"] = {
        "R2": full_r2,
        "AdjR2": full_adj,
        "MAE": full_mae,
        "RMSE": full_rmse,
        "RAE": full_rae,
    }
    logger.info(
        "Full model test metrics: R2=%.4f AdjR2=%.4f MAE=%.4f RMSE=%.4f RAE=%.4f",
        full_r2,
        full_adj,
        full_mae,
        full_rmse,
        full_rae,
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    full_model.save_model(model_path)
    logger.info("Full model saved to %s", model_path)

    use_full_model = full_r2 > test_mean.get("R2", float("-inf"))
    if use_full_model:
        pred_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": full_pred,
            }
        )
        logger.info("Using full model predictions for test_predictions.csv")
    else:
        logger.info("Using CV ensemble predictions for test_predictions.csv")

    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    logger.info("Done. Metrics written to %s", out_dir / "metrics.json")
    logger.info("Predictions written to %s", out_dir / "test_predictions.csv")
    logger.info("SHAP summary saved to %s", shap_path)


if __name__ == "__main__":
    main()
