# MLP vs. Logistic Regression — Breast Cancer Wisconsin

Binary classification of breast tumours as **malignant** or **benign** using two models, a logistic regression and a multi-layer perceptron (MLP), evaluated with standard diagnostic metrics.

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded directly from scikit-learn:

- 569 samples (212 malignant, 357 benign)
- 30 numeric features describing cell nuclei from fine-needle aspirate images (radius, texture, perimeter, area, smoothness, concavity, etc.)

**Positive class = malignant.** scikit-learn encodes malignant as 0, so the labels are flipped.

## Method

1. Stratified train/test split (75% / 25%, `random_state=42`).
2. Both models are wrapped in a pipeline with `StandardScaler`. The scaler is fit on the training data only.
3. Models:
   - **Logistic Regression**: `max_iter=1000`
   - **MLP**: hidden layers `(32, 16)`, ReLU, `alpha=1e-3`, `max_iter=2000`
4. Evaluation on the held-out test set (143 samples).

## Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Se (sensitivity) | TP / (TP + FN) | Share of cancers detected |
| Sp (specificity) | TN / (TN + FP) | Share of benign cases correctly cleared |
| PPV | TP / (TP + FP) | Reliability of a "malignant" prediction |
| NPV | TN / (TN + FN) | Reliability of a "benign" prediction |
| MCC | — | Balanced overall score in [-1, 1] |

## Results

| Model | TP | FN | FP | TN | Se | Sp | PPV | NPV | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 49 | 4 | 1 | 89 | 0.925 | 0.989 | 0.980 | 0.957 | 0.925 |
| MLP | 48 | 5 | 0 | 90 | 0.906 | 1.000 | 1.000 | 0.947 | 0.926 |

Both models perform almost identically (MCC ≈ 0.93). Logistic regression misses one fewer cancer and is easier to interpret. The MLP produces no false alarms. With only 143 test samples, a difference of one or two cases is not statistically meaningful.

## Project structure

```
.
├── bcw_mlp_logreg.py   # training + evaluation script
├── run.sh              # creates venv, installs deps, runs the script
├── README.md
└── .gitignore
```

## How to run

**With the script (Linux / macOS / Git Bash on Windows):**
```bash
chmod +x run.sh
./run.sh
```

**Manually:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install scikit-learn
python bcw_mlp_logreg.py
```

**In PyCharm:** set the project interpreter to `.venv`, install `scikit-learn`, then right-click `bcw_mlp_logreg.py` → **Run**.

## Requirements

- Python 3.10+
- scikit-learn

## Author

**GAITA MARIAN GABRIEL**
