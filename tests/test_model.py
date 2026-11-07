"""Tests for ft_linear_regression. Run with: uv run pytest"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import ModelError, estimate_price, load_dataset, load_thetas, save_thetas  # noqa: E402
from precision import compute_metrics  # noqa: E402
from train import denormalize, gradient_descent, standardize  # noqa: E402

DATA = ROOT / "data.csv"


def closed_form(xs, ys):
    """Exact least-squares line, used ONLY in tests to check gradient descent.

    theta1 = cov(x, y) / var(x), theta0 = mean(y) - theta1 * mean(x)
    """
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    theta1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - theta1 * mx, theta1


def train_on(xs, ys, lr=0.1, iterations=1000):
    scaled, mean, std = standardize(xs)
    t0, t1, history = gradient_descent(scaled, ys, lr, iterations)
    return (*denormalize(t0, t1, mean, std), history)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# ---------- hypothesis ----------

def test_hypothesis_is_theta0_plus_theta1_times_mileage():
    assert estimate_price(100, 2, 3) == 302
    assert estimate_price(5000, 0, 0) == 0


# ---------- training ----------

def test_gradient_descent_matches_closed_form_on_project_data():
    xs, ys, _ = load_dataset(DATA)
    t0, t1, _ = train_on(xs, ys)
    e0, e1 = closed_form(xs, ys)
    assert t0 == pytest.approx(e0, rel=1e-6)
    assert t1 == pytest.approx(e1, rel=1e-6)


def test_learns_a_perfect_line():
    xs = [0, 10, 20, 30, 40]
    ys = [5 + 2 * x for x in xs]
    t0, t1, _ = train_on(xs, ys)
    assert t0 == pytest.approx(5, abs=1e-6)
    assert t1 == pytest.approx(2, abs=1e-6)


def test_cost_never_increases_with_small_learning_rate():
    xs, ys, _ = load_dataset(DATA)
    *_, history = train_on(xs, ys, lr=0.1, iterations=200)
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:]))


def test_first_step_uses_subject_formula_with_simultaneous_update():
    xs, ys, lr = [1.0, 2.0, 3.0], [2.0, 4.0, 7.0], 0.01
    t0, t1, _ = gradient_descent(xs, ys, lr, 1)
    m = len(xs)
    # From theta0 = theta1 = 0, every estimate is 0, so error = -price.
    expected_t0 = -lr * (1 / m) * sum(0 - y for y in ys)
    expected_t1 = -lr * (1 / m) * sum((0 - y) * x for x, y in zip(xs, ys))
    assert t0 == pytest.approx(expected_t0)
    assert t1 == pytest.approx(expected_t1)


def test_second_step_uses_old_thetas_for_both_gradients():
    xs, ys, lr = [1.0, 2.0, 3.0], [2.0, 4.0, 7.0], 0.01
    a0, a1, _ = gradient_descent(xs, ys, lr, 1)
    m = len(xs)
    errors = [estimate_price(x, a0, a1) - y for x, y in zip(xs, ys)]
    exp0 = a0 - lr * sum(errors) / m
    exp1 = a1 - lr * sum(e * x for e, x in zip(errors, xs)) / m
    b0, b1, _ = gradient_descent(xs, ys, lr, 2)
    assert (b0, b1) == pytest.approx((exp0, exp1))


def test_too_high_learning_rate_is_reported():
    xs, ys, _ = load_dataset(DATA)
    with pytest.raises(ModelError):
        gradient_descent(xs, ys, 1.0, 1000)  # raw mileage + big rate -> diverges


def test_identical_mileages_are_rejected():
    with pytest.raises(ModelError):
        standardize([5.0, 5.0, 5.0])


# ---------- dataset loading ----------

@pytest.mark.parametrize("content", [
    "",
    "km,price\n",
    "km\n1\n",
    "km,price\n1,2,3\n",
    "km,price\nabc,3\n",
    "km,price\n1,nan\n",
    "km,price\ninf,2\n",
])
def test_bad_datasets_raise_model_error(tmp_path, content):
    with pytest.raises(ModelError):
        load_dataset(write(tmp_path, "d.csv", content))


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(ModelError):
        load_dataset(tmp_path / "nope.csv")


def test_blank_lines_are_ignored(tmp_path):
    xs, ys, names = load_dataset(write(tmp_path, "d.csv", "a,b\n\n1,2\n\n3,4\n"))
    assert (xs, ys, names) == ([1, 3], [2, 4], ("a", "b"))


# ---------- thetas file ----------

def test_missing_thetas_mean_untrained(tmp_path):
    assert load_thetas(tmp_path / "thetas.json") == (0.0, 0.0, False)


def test_thetas_round_trip(tmp_path):
    path = tmp_path / "thetas.json"
    save_thetas(path, 1.5, -0.25)
    assert load_thetas(path) == (1.5, -0.25, True)


@pytest.mark.parametrize("content", ["not json", "{}", '{"theta0": 1}',
                                     '{"theta0": "x", "theta1": 1}', "[1, 2]"])
def test_corrupt_thetas_raise(tmp_path, content):
    with pytest.raises(ModelError):
        load_thetas(write(tmp_path, "thetas.json", content))


# ---------- metrics ----------

def test_metrics_perfect_fit():
    m = compute_metrics([1, 2, 3], [3, 5, 7], 1, 2)
    assert m["mae"] == 0 and m["mse"] == 0 and m["r2"] == 1


# ---------- programs end to end ----------

def run(script, *args, stdin=""):
    return subprocess.run([sys.executable, str(ROOT / script), *args],
                          input=stdin, capture_output=True, text=True, cwd=ROOT)


def test_predict_before_training_gives_zero(tmp_path):
    result = run("predict.py", "--thetas", str(tmp_path / "t.json"), stdin="42000\n")
    assert result.returncode == 0
    assert "0.00" in result.stdout


def test_predict_reprompts_on_bad_input(tmp_path):
    result = run("predict.py", "--thetas", str(tmp_path / "t.json"),
                 stdin="abc\n-5\n\n100\n")
    assert result.returncode == 0
    assert result.stderr.count("Invalid mileage") == 3


def test_predict_handles_eof(tmp_path):
    result = run("predict.py", "--thetas", str(tmp_path / "t.json"), stdin="")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_train_then_predict(tmp_path):
    thetas = tmp_path / "t.json"
    assert run("train.py", "--output", str(thetas)).returncode == 0
    saved = json.loads(thetas.read_text())
    e0, e1 = closed_form(*load_dataset(DATA)[:2])
    assert saved["theta0"] == pytest.approx(e0, rel=1e-6)
    result = run("predict.py", "--thetas", str(thetas), stdin="100000\n")
    assert f"{e0 + e1 * 100000:.2f}" in result.stdout


def test_train_rejects_bad_arguments():
    assert run("train.py", "--learning-rate", "-1").returncode != 0
    assert run("train.py", "--iterations", "0").returncode != 0


def test_train_reports_divergence_cleanly(tmp_path):
    result = run("train.py", "--output", str(tmp_path / "t.json"),
                 "--no-normalize", "--learning-rate", "0.01")
    assert result.returncode == 1
    assert "diverged" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "t.json").exists()


def test_saved_thetas_are_finite(tmp_path):
    thetas = tmp_path / "t.json"
    run("train.py", "--output", str(thetas))
    t0, t1, _ = load_thetas(thetas)
    assert math.isfinite(t0) and math.isfinite(t1)
