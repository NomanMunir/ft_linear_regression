"""Shared code for ft_linear_regression.

This module holds the parts that both programs need:
    - the hypothesis   estimate_price(mileage) = theta0 + (theta1 * mileage)
    - reading the dataset (CSV with a header and two numeric columns)
    - saving and loading theta0 / theta1

Only the Python standard library is used here.
"""

import csv
import json
import math
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data.csv"
DEFAULT_THETAS_PATH = PROJECT_DIR / "thetas.json"


class ModelError(Exception):
    """Raised for any user-facing problem (bad file, bad value, ...)."""


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """The hypothesis given by the subject, written exactly as specified."""
    return theta0 + (theta1 * mileage)


def parse_number(text: str) -> float:
    """Convert text to a finite float, or raise ModelError."""
    try:
        value = float(text.strip())
    except ValueError:
        raise ModelError(f"'{text.strip()}' is not a number") from None
    if not math.isfinite(value):
        raise ModelError(f"'{text.strip()}' is not a finite number")
    return value


def load_dataset(path: Path) -> tuple[list[float], list[float], tuple[str, str]]:
    """Read a CSV file with a header line and two numeric columns.

    The first column is the feature (mileage), the second the target (price).
    Column names are not hard-coded, so any two-column dataset works.

    Returns (mileages, prices, (feature_name, target_name)).
    """
    try:
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
    except FileNotFoundError:
        raise ModelError(f"dataset not found: {path}") from None
    except IsADirectoryError:
        raise ModelError(f"dataset is a directory: {path}") from None
    except PermissionError:
        raise ModelError(f"cannot read dataset (permission denied): {path}") from None
    except (UnicodeDecodeError, csv.Error) as error:
        raise ModelError(f"cannot parse dataset {path}: {error}") from None

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise ModelError(f"dataset is empty: {path}")

    header = rows[0]
    if len(header) != 2:
        raise ModelError(f"header must have exactly 2 columns, got {len(header)}")

    mileages: list[float] = []
    prices: list[float] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ModelError(f"line {line_number}: expected 2 columns, got {len(row)}")
        try:
            mileages.append(parse_number(row[0]))
            prices.append(parse_number(row[1]))
        except ModelError as error:
            raise ModelError(f"line {line_number}: {error}") from None

    if not mileages:
        raise ModelError(f"dataset has a header but no data rows: {path}")
    return mileages, prices, (header[0].strip(), header[1].strip())


def load_thetas(path: Path) -> tuple[float, float, bool]:
    """Load theta0 and theta1.

    If the file does not exist the model is untrained, so both thetas are 0,
    as the subject requires. Returns (theta0, theta1, trained).
    """
    if not path.exists():
        return 0.0, 0.0, False
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ModelError(f"cannot read thetas from {path}: {error!r}") from None
    if not (math.isfinite(theta0) and math.isfinite(theta1)):
        raise ModelError(f"thetas in {path} are not finite numbers")
    return theta0, theta1, True


def save_thetas(path: Path, theta0: float, theta1: float) -> None:
    """Save the thetas. The write goes to a temp file first, then replaces the
    real file, so a crash never leaves a half-written thetas file."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump({"theta0": theta0, "theta1": theta1}, file, indent=2)
            file.write("\n")
        os.replace(tmp_path, path)
    except OSError as error:
        raise ModelError(f"cannot write thetas to {path}: {error}") from None
