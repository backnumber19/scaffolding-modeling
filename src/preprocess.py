import argparse
from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import setup_logger


def load_raw(input_path: Path, target_col: str, index_col: str, logger: logging.Logger):
    df = pd.read_excel(input_path)
    if index_col in df.columns:
        df = df.set_index(index_col)
    else:
        logger.warning("Index column %s not found; using default index.", index_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    # Drop duplicated index rows
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %s duplicated index rows.", dropped)

    # Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        logger.info("Dropped %s empty columns.", len(empty_cols))

    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="data/FortillsDataset_JW_cleaned.xlsx"
    )
    parser.add_argument("--target-col", type=str, default="SumOfManhoursProrate")
    parser.add_argument("--index-col", type=str, default="TaskID")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, log_name="preprocess.log")

    logger.info(
        "Preprocess start: input=%s target=%s index=%s test_size=%s seed=%s",
        input_path,
        args.target_col,
        args.index_col,
        args.test_size,
        args.seed,
    )

    X, y = load_raw(input_path, args.target_col, args.index_col, logger)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    logger.info(
        "Split shapes: X_train=%s X_test=%s y_train=%s y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    train_target = pd.DataFrame({args.target_col: y_train.values}, index=X_train.index)
    test_target = pd.DataFrame({args.target_col: y_test.values}, index=X_test.index)

    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()
    X_train_raw.insert(0, args.index_col, X_train.index)
    X_test_raw.insert(0, args.index_col, X_test.index)
    X_train_raw.to_excel(out_dir / "train.xlsx", index=False)
    X_test_raw.to_excel(out_dir / "test.xlsx", index=False)
    train_target.to_excel(out_dir / "train_target.xlsx", index=False)
    test_target.to_excel(out_dir / "test_target.xlsx", index=False)

    logger.info("Saved train/test features and targets to %s", out_dir)
    logger.info("Preprocess complete.")


if __name__ == "__main__":
    main()
