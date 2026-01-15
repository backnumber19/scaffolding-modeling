import numpy as np
from pathlib import Path
import logging


def adj_r2(r2, n, p):
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def rae(y_true, y_pred):
    y_mean = np.mean(y_true)
    denom = np.sum(np.abs(y_true - y_mean))
    if denom == 0:
        return float("nan")
    return np.sum(np.abs(y_true - y_pred)) / denom


def setup_logger(log_dir: Path, log_name: str = "train.log") -> logging.Logger:
    """
    Configure a root logger that writes to file and stdout.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logger = logging.getLogger("scaffold_train")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
