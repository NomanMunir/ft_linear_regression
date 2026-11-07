# ft_linear_regression
# Usage: make [all|train|predict|plot|precision|test|clean|fclean|re|help]
# Extra arguments: make train ARGS="--learning-rate 0.5 --iterations 200"

UV      := uv
RUN     := $(UV) run
VENV    := .venv
THETAS  := thetas.json
ARGS    ?=

# ---------- standard 42 rules ----------

all: $(VENV)

# Rebuilt only when the dependency files change (no relink).
$(VENV): pyproject.toml uv.lock
	$(UV) sync
	@touch $(VENV)

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache $(THETAS).tmp *.png

fclean: clean
	rm -rf $(VENV) $(THETAS)

re: fclean all

# ---------- mandatory ----------

train: $(VENV)
	$(RUN) train.py $(ARGS)

predict: $(VENV)
	$(RUN) predict.py $(ARGS)

# ---------- bonus ----------

plot: $(VENV)
	$(RUN) plot.py $(ARGS)

precision: $(VENV)
	$(RUN) precision.py $(ARGS)

# ---------- extra ----------

test: $(VENV)
	$(RUN) pytest -v

help:
	@echo "make            install dependencies (uv sync)"
	@echo "make train      train the model, writes $(THETAS)"
	@echo "make predict    prompt for a mileage and estimate its price"
	@echo "make plot       bonus: plot the data and the regression line"
	@echo "make precision  bonus: print MAE, MSE, RMSE and R^2"
	@echo "make test       run the test suite"
	@echo "make clean      remove caches and temporary files"
	@echo "make fclean     clean + remove .venv and $(THETAS) (model back to untrained)"
	@echo "make re         fclean + all"
	@echo 'Pass options with ARGS, e.g. make train ARGS="--learning-rate 0.5"'

.PHONY: all clean fclean re train predict plot precision test help
