"""
MLP vs. Logistic Regression on Breast Cancer Wisconsin (Diagnostic).

Predicts malignant vs. benign tumours from 30 cell-nucleus features
(569 samples) and reports: confusion matrix, Se, Sp, PPV, NPV, MCC.

Positive class = malignant (sklearn uses 0 = malignant, so we flip it).
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef

RANDOM_STATE = 42   # reproducibility
TEST_SIZE = 0.25    # 25% held out for testing


def load_data():
    """Load the dataset; return X (569x30) and y (1 = malignant, 0 = benign)."""
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, "
          f"malignant: {y.sum()}, benign: {(y == 0).sum()}")
    return X, y


def build_models():
    """
    Return the two models, each with a StandardScaler in a pipeline
    (scaler is fit on training data only -> no test leakage).
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                          alpha=1e-3, max_iter=2000,
                          random_state=RANDOM_STATE)),
    }


def compute_metrics(y_true, y_pred):
    """
    Return confusion-matrix counts and metrics:
    Se = TP/(TP+FN), Sp = TN/(TN+FP), PPV = TP/(TP+FP),
    NPV = TN/(TN+FN), MCC in [-1, 1].
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Se (sensitivity)": tp / (tp + fn),
        "Sp (specificity)": tn / (tn + fp),
        "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
        "NPV": tn / (tn + fn) if (tn + fn) else float("nan"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def print_report(name, m):
    """Print the confusion matrix and metrics for one model."""
    print(f"\n=== {name} ===")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                  Pred. malignant  Pred. benign")
    print(f"Actual malignant  {m['TP']:>15}  {m['FN']:>12}")
    print(f"Actual benign     {m['FP']:>15}  {m['TN']:>12}")
    for k in ["Se (sensitivity)", "Sp (specificity)", "PPV", "NPV", "MCC"]:
        print(f"{k:<18}: {m[k]:.4f}")


def main():
    """Stratified train/test split, fit each model, evaluate on test set."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        print_report(name, compute_metrics(y_test, model.predict(X_test)))


if __name__ == "__main__":
    main()