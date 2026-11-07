#!/usr/bin/env python3
"""Program 2: train the model with gradient descent and save theta0, theta1.

The training step is the one given in the subject:

    tmp_theta0 = learningRate * (1/m) * sum(estimatePrice(mileage[i]) - price[i])
    tmp_theta1 = learningRate * (1/m) * sum((estimatePrice(mileage[i]) - price[i]) * mileage[i])

and both thetas are updated at the same time (simultaneous update).

Feature scaling: mileages go up to ~240 000 while prices are ~3 000 to 8 000.
On raw data the theta1 gradient is about 10^10 times larger than the theta0
gradient, so no single learning rate works for both. We standardize the
mileage first (x' = (x - mean) / std), run the exact same gradient descent on
x', then convert the thetas back so predict.py can use raw kilometres.
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
    save_thetas,
)


class DivergenceError(ModelError):
    """Raised when the thetas blow up (learning rate too high)."""


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Mean and (population) standard deviation."""
    m = len(values)
    mean = sum(values) / m
    variance = sum((v - mean) ** 2 for v in values) / m
    return mean, math.sqrt(variance)


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    """Return values rescaled to mean 0 and standard deviation 1."""
    mean, std = mean_and_std(values)
    if std == 0:
        raise ModelError("all mileages are identical: a line cannot be fitted")
    return [(v - mean) / std for v in values], mean, std


def cost(mileages: list[float], prices: list[float], theta0: float, theta1: float) -> float:
    """Mean squared error divided by 2: J = 1/(2m) * sum(error^2).

    Gradient descent minimizes this function. The subject formulas are its
    partial derivatives (see docs/concepts.html).
    """
    m = len(mileages)
    total = 0.0
    for x, y in zip(mileages, prices):
        error = estimate_price(x, theta0, theta1) - y
        # error * error (not error ** 2): on overflow it gives inf instead of
        # raising OverflowError, so divergence is detected below cleanly.
        total += error * error
    return total / (2 * m)


def gradient_descent(
    mileages: list[float],
    prices: list[float],
    learning_rate: float,
    iterations: int,
) -> tuple[float, float, list[float]]:
    """Run gradient descent from theta0 = theta1 = 0.

    Returns (theta0, theta1, cost_history). The history holds the cost
    before each step, then the final cost.
    """
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0
    history = [cost(mileages, prices, theta0, theta1)]

    for _ in range(iterations):
        sum_errors = 0.0
        sum_errors_times_mileage = 0.0
        for i in range(m):
            error = estimate_price(mileages[i], theta0, theta1) - prices[i]
            sum_errors += error
            sum_errors_times_mileage += error * mileages[i]

        tmp_theta0 = learning_rate * (1 / m) * sum_errors
        tmp_theta1 = learning_rate * (1 / m) * sum_errors_times_mileage

        # Simultaneous update: both tmp values above were computed with the
        # OLD theta0 and theta1. Only now do we change them.
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        current_cost = cost(mileages, prices, theta0, theta1)
        if not (math.isfinite(theta0) and math.isfinite(theta1) and math.isfinite(current_cost)):
            raise DivergenceError(
                f"training diverged (learning rate {learning_rate} is too high)"
            )
        history.append(current_cost)

    return theta0, theta1, history


def denormalize(theta0_n: float, theta1_n: float, mean: float, std: float) -> tuple[float, float]:
    """Convert thetas learned on x' = (x - mean) / std back to raw x.

    price = theta0_n + theta1_n * (x - mean) / std
          = (theta0_n - theta1_n * mean / std) + (theta1_n / std) * x
    """
    theta1 = theta1_n / std
    theta0 = theta0_n - theta1_n * mean / std
    return theta0, theta1


def positive_float(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the car price model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="CSV dataset (default: data.csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_THETAS_PATH,
                        help="where to save the thetas (default: thetas.json)")
    parser.add_argument("--learning-rate", type=positive_float, default=0.1,
                        help="gradient descent step size (default: 0.1)")
    parser.add_argument("--iterations", type=positive_int, default=1000,
                        help="number of gradient descent steps (default: 1000)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="train on raw mileage (for experiments: needs a tiny "
                             "learning rate and converges very slowly)")
    args = parser.parse_args()

    try:
        mileages, prices, _ = load_dataset(args.data)
        if len(mileages) < 2:
            raise ModelError("need at least 2 data rows to fit a line")

        if args.no_normalize:
            theta0, theta1, history = gradient_descent(
                mileages, prices, args.learning_rate, args.iterations)
        else:
            scaled, mean, std = standardize(mileages)
            theta0_n, theta1_n, history = gradient_descent(
                scaled, prices, args.learning_rate, args.iterations)
            theta0, theta1 = denormalize(theta0_n, theta1_n, mean, std)

        save_thetas(args.output, theta0, theta1)
    except ModelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Trained on {len(mileages)} rows, {args.iterations} iterations, "
          f"learning rate {args.learning_rate}"
          f"{' (raw mileage)' if args.no_normalize else ''}.")
    print(f"Cost: {history[0]:.2f} -> {history[-1]:.2f}")
    print(f"theta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.10f}")
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
