import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


RANDOM_STATE = 42
MISSING_RATE = 0.10
DATASET_PATH = "heart.csv"


def load_dataset(path):
    """
    Loads the dataset from a CSV file.

    Parameters:
        path (str): Path to the dataset file.

    Returns:
        X (DataFrame): Feature columns.
        y (Series): Target column.
        df (DataFrame): Full dataset.
    """
    df = pd.read_csv(path)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y, df


def check_missing_values(df):
    """
    Checks missing values in the dataset.

    Parameters:
        df (DataFrame): Full dataset.

    Returns:
        Series: Number of missing values per column.
    """
    return df.isnull().sum()


def create_artificial_missing_values(X, missing_rate=0.10, random_state=42):
    """
    Creates artificial missing values in the feature columns.

    This is used because the original dataset does not contain missing values.
    The target column is not modified.

    Parameters:
        X (DataFrame): Feature columns.
        missing_rate (float): Percentage of values to replace with NaN.
        random_state (int): Random seed for reproducibility.

    Returns:
        DataFrame: Feature dataset with artificial missing values.
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.astype(float).copy()

    mask = rng.random(X_missing.shape) < missing_rate
    X_missing = X_missing.mask(mask)

    return X_missing


def build_model(imputer):
    """
    Builds a machine learning pipeline.

    The pipeline first applies imputation and then trains a Gradient Boosting model.

    Parameters:
        imputer: Imputation method, for example SimpleImputer or KNNImputer.

    Returns:
        Pipeline: Complete machine learning pipeline.
    """
    model = Pipeline([
        ("imputer", imputer),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the model.

    Parameters:
        model (Pipeline): Machine learning pipeline.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.

    Returns:
        dict: Evaluation results.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return results


def run_experiment():
    
    X, y, df = load_dataset(DATASET_PATH)

    print("Dataset shape:", df.shape)
    print("\nMissing values in the original dataset:")
    print(check_missing_values(df))

    X_missing = create_artificial_missing_values(
        X,
        missing_rate=MISSING_RATE,
        random_state=RANDOM_STATE
    )

    print("\nMissing values after artificial missing data:")
    print(X_missing.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_missing,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    imputation_methods = {
        "Mean Imputation": SimpleImputer(strategy="mean"),
        "KNN Imputation": KNNImputer(n_neighbors=5)
    }

    all_results = []

    for method_name, imputer in imputation_methods.items():
        model = build_model(imputer)
        results = evaluate_model(model, X_train, X_test, y_train, y_test)

        all_results.append({
            "Imputation Method": method_name,
            "Accuracy": round(results["Accuracy"], 3),
            "F1-score": round(results["F1-score"], 3),
            "ROC-AUC": round(results["ROC-AUC"], 3),
            "Confusion Matrix": results["Confusion Matrix"].tolist()
        })

    results_df = pd.DataFrame(all_results)

    print("\nGradient Boosting Results with Different Imputation Methods:")
    print(results_df.to_string(index=False))

    results_df.to_csv("person4_results.csv", index=False)
    print("\nResults saved as person4_results.csv")


if __name__ == "__main__":
    run_experiment()