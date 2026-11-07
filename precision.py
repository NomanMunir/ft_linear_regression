#!/usr/bin/env python3
"""Bonus: measure how precise the trained model is on the dataset.

Metrics (all computed by hand, no library):
    MAE   mean absolute error        average size of a mistake, in price units
    MSE   mean squared error         punishes big mistakes more
    RMSE  root mean squared error    MSE back in price units
    R^2   coefficient of determination
          1.0 = perfect, 0.0 = no better than always predicting the mean price
"""

import argparse
import math
import sys
from pathlib import Path

from model import (
    DEFAULT_DATA_PATH,
    DEFAULT_THETAS_PATH,
    ModelError,
    estimate_price,
    load_dataset,
    load_thetas,
)


def compute_metrics(
    mileages: list[float], prices: list[float], theta0: float, theta1: float
) -> dict[str, float]:
    m = len(mileages)
    errors = [estimate_price(x, theta0, theta1) - y for x, y in zip(mileages, prices)]
    mae = sum(abs(e) for e in errors) / m
    mse = sum(e * e for e in errors) / m
    mean_price = sum(prices) / m
    total_variation = sum((y - mean_price) ** 2 for y in prices)
    if total_variation == 0:
        r2 = 1.0 if mse == 0 else float("-inf")
    else:
        r2 = 1 - (mse * m) / total_variation
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "r2": r2,
            "mean_price": mean_price}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--thetas", type=Path, default=DEFAULT_THETAS_PATH)
    args = parser.parse_args()

    try:
        mileages, prices, _ = load_dataset(args.data)
        theta0, theta1, trained = load_thetas(args.thetas)
    except ModelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not trained:
        print("Note: model not trained yet (theta0 = theta1 = 0). Run train.py first.")

    metrics = compute_metrics(mileages, prices, theta0, theta1)
    print(f"Samples          : {len(mileages)}")
    print(f"theta0, theta1   : {theta0:.6f}, {theta1:.10f}")
    print(f"MAE              : {metrics['mae']:.2f}")
    print(f"MSE              : {metrics['mse']:.2f}")
    print(f"RMSE             : {metrics['rmse']:.2f}")
    print(f"R^2              : {metrics['r2']:.4f}")
    print(f"Mean error vs mean price: "
          f"{100 * metrics['mae'] / metrics['mean_price']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
