# Tabular Examples

Differentially private synthetic **tabular** data with Private Evolution (PE). Each script downloads a public dataset (train/test split + metadata), runs PE, and writes results to `results/tabular/<experiment>/`:

- `synthetic_tab/` — generated synthetic CSVs
- `checkpoint/` — per-iteration checkpoints (runs resume from here)
- classifier accuracy (`TabClassifier`) and Wasserstein-style marginal distance (`ComputeWSD`) logged to `log.txt` and CSV


*Our code automatically downloads the datasets, which are available at https://github.com/toan-vt/cloud-data-store/tree/main/tabular.*

## Installation

```bash
pip install private-evolution[tabular]
```

Or, for an editable install from the repo root:

```bash
pip install -e ".[tabular]"
```

## Simulated / stress tests

XOR datasets with a configurable number of features (from 1 to 7):

```bash
python xor_stress_test.py --num-features 1
python xor_stress_test.py --num-features 2
```

Structural causal model (SCM) data with a selectable prior function:

```bash
python scm.py --prior-function rff   # choices: tree, nn, rff
```

## Real datasets

```bash
python artificial_characters.py
python person_activity.py
python adult.py
python breast_cancer.py
```