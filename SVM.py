import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# 1. HELPER FUNCTION (MEAN + STD)
def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


# 2. MODEL EVALUATION FUNCTION (CV + TRAIN/TEST)
def evaluate_binary_model(name, pipeline, X_train, y_train, X_test, y_test, cv):

    # cross-validation scoring setup
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

    # 2.1 CROSS VALIDATION
    cv_res = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=None,
    )

    # compute CV mean/std
    cv_accuracy_mean, cv_accuracy_std = mean_std(cv_res["test_accuracy"])
    cv_f1_mean, cv_f1_std = mean_std(cv_res["test_f1"])
    cv_roc_auc_mean, cv_roc_auc_std = mean_std(cv_res["test_roc_auc"])

    # 2.2 TRAIN MODEL ON FULL TRAIN SET
    pipeline.fit(X_train, y_train)

    # predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # 2.3 TRAIN / TEST METRICS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    train_roc_auc = roc_auc_score(y_train, train_proba)
    test_roc_auc = roc_auc_score(y_test, test_proba)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    return {
        "model": name,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std": cv_accuracy_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "confusion_matrix": cm,
    }


# 3. MAIN PIPELINE
def main():

    # 3.1 LOAD DATA
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        print("Error: heart.csv not found.")
        sys.exit(1)

    # 3.2 CHECK TARGET COLUMN
    if "target" not in df.columns:
        print("Error: dataset must contain 'target'.")
        sys.exit(1)

    # split X and y
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3.3 VALIDATE TARGET
    if y.nunique() != 2:
        print("Error: target must be binary.")
        sys.exit(1)

    # encode target if needed
    if not set(pd.unique(y)).issubset({0, 1, True, False}):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = pd.Series(y).astype(int).values

    # 3.4 TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3.5 CROSS VALIDATION SETUP
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3.6 MODELS TO TEST
    models = {
        "SVM Linear": SVC(kernel="linear", probability=True, random_state=42),
        "SVM RBF": SVC(kernel="rbf", probability=True, random_state=42),
    }

    # 3.7 TRAIN + EVALUATE MODELS
    results = []

    for name, svm in models.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm)
        ])

        results.append(
            evaluate_binary_model(
                name=name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
            )
        )

    # 3.8 CREATE RESULTS TABLE
    results_df = pd.DataFrame([
        {
            "Model": r["model"],
            "CV Accuracy (mean)": r["cv_accuracy_mean"],
            "CV Accuracy (std)": r["cv_accuracy_std"],
            "CV F1 (mean)": r["cv_f1_mean"],
            "CV ROC-AUC (mean)": r["cv_roc_auc_mean"],
            "Train Accuracy": r["train_accuracy"],
            "Test Accuracy": r["test_accuracy"],
            "Train F1": r["train_f1"],
            "Test F1": r["test_f1"],
            "Train ROC-AUC": r["train_roc_auc"],
            "Test ROC-AUC": r["test_roc_auc"],
        }
        for r in results
    ])

    # round numeric values
    numeric_cols = [c for c in results_df.columns if results_df[c].dtype.kind in {"i", "u", "f"}]
    results_df[numeric_cols] = results_df[numeric_cols].round(4)

    print("FINAL RESULTS TABLE")
    print(results_df.to_string(index=False))
    print()

    # 3.9 CONFUSION MATRICES
    print("CONFUSION MATRICES")
    for r in results:
        cm_df = pd.DataFrame(
            r["confusion_matrix"],
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"]
        )
        print(r["model"])
        print(cm_df)
        print()

    # 3.10 BEST MODEL BY CV
    best_cv = sorted(
        results,
        key=lambda x: (x["cv_roc_auc_mean"], x["cv_f1_mean"], x["cv_accuracy_mean"]),
        reverse=True,
    )[0]

    # 3.11 STABILITY CHECK
    stability = sorted(
        results,
        key=lambda x: np.mean([
            x["cv_accuracy_std"],
            x["cv_f1_std"],
            x["cv_roc_auc_std"]
        ]),
    )[0]

    # 3.12 OVERFITTING ANALYSIS
    def overfitting_gap(train_val, test_val):
        return float(train_val - test_val)

    gaps = []
    for r in results:
        gaps.append(
            (
                r["model"],
                overfitting_gap(r["train_accuracy"], r["test_accuracy"]),
                overfitting_gap(r["train_f1"], r["test_f1"]),
                overfitting_gap(r["train_roc_auc"], r["test_roc_auc"]),
            )
        )

    most_overfit = sorted(gaps, key=lambda t: (t[1], t[3]), reverse=True)[0]

    # 3.13 PRINT INTERPRETATION
    print("INTERPRETATION")

    print(f"Best CV model: {best_cv['model']} (ROC-AUC={best_cv['cv_roc_auc_mean']:.4f})")

    print(
        f"Most stable model: {stability['model']} "
        f"(avg CV std={np.mean([stability['cv_accuracy_std'], stability['cv_f1_std'], stability['cv_roc_auc_std']]):.4f})"
    )

    print(
        f"Most overfitting candidate: {most_overfit[0]} "
        f"(accuracy gap={most_overfit[1]:.4f}, ROC-AUC gap={most_overfit[3]:.4f})"
    )


# 4. RUN PROGRAM
if __name__ == "__main__":
    main()