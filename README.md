# Assessment Design

Numerical examples and independent verification routines for synthetic assessment,
capacity, recovery, and shared-budget training models. Inputs are synthetic; no
external datasets are required.

## Setup

Tested with Python 3.9.6 and the package versions in requirements.txt.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Run the examples

Run from the repository directory:

```sh
.venv/bin/python numerics.py
.venv/bin/python training_comparison.py
.venv/bin/python policy_illustrations.py
```

The scripts generate plots, numerical outputs, and verification reports locally.
Existing JSON and CSV outputs are overwritten when regenerated. Plot and report
directories are excluded by .gitignore. LP solver versions may produce small
differences in optimal mixture coordinates; numerical checks use tolerances.

## Run independent checks

```sh
.venv/bin/python checks/verify_theory.py
.venv/bin/python checks/verify_training.py
.venv/bin/python checks/verify_training_threshold.py
.venv/bin/python checks/verify_policy_revision.py
```

## Files

- `numerics.py`: assessment frontiers, capacity dynamics, recovery, assistance
  scenarios, random-technology experiments, and numerical consistency checks.
- `training_comparison.py`: critical-pressure calculations and finite-candidate
  optimization under a shared resource budget.
- `policy_illustrations.py`: simple-policy comparisons, trajectories, reward
  constraints, and sensitivity calculations.
- `checks/`: independent allocation LPs, dense-search comparisons, threshold
  verification, and candidate-supply checks.
- `*.json`, `*.csv`: synthetic parameters and unrounded reference outputs.
