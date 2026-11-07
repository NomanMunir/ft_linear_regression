# ft_linear_regression

An introduction to machine learning (42 school project): predict the price of a car from its mileage with a linear function trained by gradient descent.

```
estimatePrice(mileage) = θ0 + (θ1 * mileage)
```

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.10+ (uv can install it for you)

```sh
uv sync
```

## Usage

### Mandatory part

```sh
uv run predict.py   # before training θ0 = θ1 = 0, so the estimate is 0
uv run train.py     # gradient descent on data.csv, writes thetas.json
uv run predict.py   # prompts for a mileage and prints the estimated price
```

Example:

```
$ uv run train.py
Trained on 24 rows, 1000 iterations, learning rate 0.1.
Cost: 20880519.29 -> 222822.62
theta0 = 8499.599650
theta1 = -0.0214489636
Saved to .../thetas.json

$ uv run predict.py
Enter a mileage (km): 100000
Estimated price for 100000 km: 6354.70
```

`train.py` options: `--data FILE`, `--output FILE`, `--learning-rate α` (default 0.1), `--iterations N` (default 1000), `--no-normalize` (for experiments).

### Bonus

```sh
uv run plot.py                 # data + regression line (add --save plot.png to write a file)
uv run precision.py            # MAE, MSE, RMSE, R²
```

### Tests

```sh
uv run pytest -v
```

## Project layout

| File | Part | Role |
|---|---|---|
| `predict.py` | mandatory | Prompts for a mileage, prints the estimate |
| `train.py` | mandatory | Gradient descent, saves θ0 and θ1 |
| `model.py` | mandatory | Hypothesis, CSV reading, theta file I/O |
| `plot.py` | bonus | Graph of the data and the line |
| `precision.py` | bonus | Precision metrics |
| `tests/` | extra | Automatic checks (incl. comparison with the exact closed-form solution) |
| `docs/` | extra | Study guide (HTML) |

## How it follows the subject

- **No library does the work.** `train.py`, `predict.py` and `model.py` use only the Python standard library. matplotlib is used only in the bonus `plot.py`, to draw.
- **The specified hypothesis.** `estimate_price` in `model.py` is `theta0 + (theta1 * mileage)`. Both programs use it.
- **The specified training function.** `gradient_descent` in `train.py` computes `tmp_theta0` and `tmp_theta1` exactly as in the subject, then updates both thetas simultaneously.
- **θ0 = θ1 = 0 before training.** `thetas.json` is generated, not committed. When it is missing, `predict.py` uses 0 and 0.

### Feature scaling

Mileage values (~10⁵) are much larger than prices (~10³), so the θ1 gradient is about 100 000× bigger than the θ0 gradient and no single learning rate fits both. `train.py` therefore standardizes the mileage (`x' = (x − mean) / std`), runs the subject's gradient descent unchanged on `x'`, and converts the result back:

```
θ1 = θ1' / std
θ0 = θ0' − θ1' · mean / std
```

`predict.py` then uses the subject's hypothesis on raw kilometres. Run `uv run train.py --no-normalize --learning-rate 1e-10` to see what happens without scaling.

## Results on `data.csv`

| | |
|---|---|
| θ0 | 8499.60 |
| θ1 | −0.0214490 |
| MAE | 557.84 |
| RMSE | 667.57 |
| R² | 0.733 |

These match the exact least-squares solution to 6 significant digits (checked in the tests).

## Study guide

Open [`docs/index.html`](docs/index.html) in a browser. It contains:

- a 7-session **study plan** with progress tracking
- **Concepts**: hypothesis, cost function, gradient descent derivation, feature scaling, metrics
- a **code walkthrough** of every file
- an interactive **playground** that runs gradient descent step by step on the real data
- **defense prep**: the evaluation checklist, likely questions with answers, and a rebuild-from-scratch challenge

The docs work offline; no server is needed.

## AI usage

The 42 subject asks for transparency about AI use. This project was built with the help of Claude (Anthropic), an AI coding assistant:

- **AI-generated:** the first version of the code (`model.py`, `train.py`, `predict.py`, `precision.py`, `plot.py`), the tests, and the study guide in `docs/`.
- **Reviewed and understood by me:** every file, following the study plan in `docs/`, including rewriting the training program from scratch without the AI's code.
- **Decisions I can defend:** standardizing the mileage, the JSON theta file, and the input and error handling. Each is explained in `docs/code.html` and `docs/defense.html`.

> Update this section to describe accurately what you changed, reviewed and learned before your defense.
