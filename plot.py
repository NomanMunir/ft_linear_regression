#!/usr/bin/env python3
"""Bonus: plot the dataset and the regression line.

matplotlib only draws points and a line here. The line itself comes from the
thetas that train.py computed with our own gradient descent.
"""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the data and the regression line.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--thetas", type=Path, default=DEFAULT_THETAS_PATH)
    parser.add_argument("--save", type=Path, metavar="FILE.png",
                        help="save the figure to a file instead of opening a window")
    args = parser.parse_args()

    try:
        mileages, prices, (x_name, y_name) = load_dataset(args.data)
        theta0, theta1, trained = load_thetas(args.thetas)
    except ModelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(mileages, prices, color="tab:blue", label="data")

    # A straight line only needs its two end points.
    x_min, x_max = min(mileages), max(mileages)
    line_x = [x_min, x_max]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]
    label = (f"price = {theta0:.2f} + ({theta1:.6f} * {x_name})" if trained
             else "untrained (theta0 = theta1 = 0)")
    ax.plot(line_x, line_y, color="tab:red", linewidth=2, label=label)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("ft_linear_regression")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
