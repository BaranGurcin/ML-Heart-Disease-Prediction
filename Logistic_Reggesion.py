import pandas as pd
import numpy as np


# 1. DATA LOAD

df = pd.read_csv("heart.csv", sep=r'\s+', encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

X = df.drop("target", axis=1)
y = df["target"]


# 2. TRAIN / TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)


# 4. MODEL FUNCTION

from sklearn.linear_model import LogisticRegression

def make_model():
    return LogisticRegression(
        max_iter=3000,
        solver='liblinear'   # daha stabil, warning azaltır
    )


# 5. METRICS FUNCTION

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    )


# 6. BASELINE MODEL

lr_base = make_model()
lr_base.fit(X_train, y_train)

acc_b, f1_b, roc_b = evaluate(lr_base, X_test, y_test)


# 7. STABILITY ANALYSIS

def stability(model_fn, X, y, n=20):
    scores = []

    for _ in range(n):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y
        )

        model = model_fn()
        model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        scores.append(accuracy_score(y_te, pred))

    return np.mean(scores), np.std(scores)

mean_b, std_b = stability(make_model, X, y)


# 8. FEATURE SELECTION - MI

from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X_train_df, y_train)
mi = pd.Series(mi, index=X.columns)

top5_mi = mi.sort_values(ascending=False).head(5).index

X_train_mi = X_train_df[top5_mi]
X_test_mi = X_test_df[top5_mi]

lr_mi = make_model()
lr_mi.fit(X_train_mi, y_train)

acc_mi, f1_mi, roc_mi = evaluate(lr_mi, X_test_mi, y_test)

# 9. FEATURE SELECTION - RFE

from sklearn.feature_selection import RFE

rfe_model = make_model()

rfe = RFE(rfe_model, n_features_to_select=5)
rfe.fit(X_train, y_train)

selected_rfe = X.columns[rfe.support_]

X_train_rfe = X_train_df[selected_rfe]
X_test_rfe = X_test_df[selected_rfe]

lr_rfe = make_model()
lr_rfe.fit(X_train_rfe, y_train)

acc_rfe, f1_rfe, roc_rfe = evaluate(lr_rfe, X_test_rfe, y_test)

# 10. FEATURE IMPORTANCE (MI)

print("\nFEATURE IMPORTANCE (Mutual Information)")
print(mi.sort_values(ascending=False))


# 11. IMPROVEMENT FUNCTION

def improvement(base, new):
    return ((new - base) / base) * 100


# 12. FINAL TABLE

results = pd.DataFrame([
    ["Baseline", acc_b, f1_b, roc_b],
    ["Mutual Information", acc_mi, f1_mi, roc_mi],
    ["RFE", acc_rfe, f1_rfe, roc_rfe]
], columns=["Method", "Accuracy", "F1", "ROC-AUC"])

results["Improvement (%)"] = [
    0,
    improvement(acc_b, acc_mi),
    improvement(acc_b, acc_rfe)
]

print("\nFINAL COMPARISON TABLE")
print(results)


# 13. STABILITY RESULT

print("\nSTABILITY ANALYSIS (BASELINE)")
print("Mean Accuracy:", mean_b)
print("Std Dev:", std_b)

