#!/usr/bin/env python3
"""Program 1: predict the price of a car for a given mileage.

It prompts for a mileage and prints
    estimate_price(mileage) = theta0 + (theta1 * mileage)
using the thetas saved by train.py. Before training, theta0 = theta1 = 0.
"""

import argparse
import sys
from pathlib import Path

from model import (
    DEFAULT_THETAS_PATH,
    ModelError,
    estimate_price,
    load_thetas,
    parse_number,
)


def ask_mileage() -> float | None:
    """Prompt until we get a valid mileage. Returns None on EOF (Ctrl-D)."""
    while True:
        try:
            text = input("Enter a mileage (km): ")
        except EOFError:
            print()
            return None
        try:
            mileage = parse_number(text)
        except ModelError as error:
            print(f"Invalid mileage: {error}. Please try again.", file=sys.stderr)
            continue
        if mileage < 0:
            print("Invalid mileage: it cannot be negative. Please try again.", file=sys.stderr)
            continue
        return mileage


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a car price from its mileage.")
    parser.add_argument(
        "--thetas", type=Path, default=DEFAULT_THETAS_PATH,
        help="file written by train.py (default: thetas.json)",
    )
    args = parser.parse_args()

    try:
        theta0, theta1, trained = load_thetas(args.thetas)
    except ModelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not trained:
        print("Note: model not trained yet, using theta0 = 0 and theta1 = 0.")

    try:
        mileage = ask_mileage()
    except KeyboardInterrupt:
        print()
        return 130
    if mileage is None:
        return 1

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price for {mileage:g} km: {price:.2f}")
    if price < 0:
        print("Warning: a negative price means this mileage is far outside "
              "the data the model learned from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
