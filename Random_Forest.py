import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


# 1. LOAD DATASET
df = pd.read_csv("heart-1.csv")

# split features and target
X = df.drop("target", axis=1)
y = df["target"]


# 2. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. STORE RESULTS
results = []


# 4. HYPERPARAMETER TUNING LOOP
for n in [50, 100, 200]:          # number of trees
    for depth in [None, 5, 10]:   # tree depth

        # create model with current parameters
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=depth,
            random_state=42
        )

        # train model
        model.fit(X_train, y_train)

        # predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # accuracy scores
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # probability-based metric
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)

        # save results
        results.append({
            "n_estimators": n,
            "max_depth": depth,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "roc_auc": roc
        })


# 5. RESULTS TABLE
results_df = pd.DataFrame(results)
print(results_df)


# 6. BEST MODEL SELECTION
best = results_df.sort_values(by="test_acc", ascending=False).iloc[0]

print("\nBEST PARAMETERS:")
print(best)


# 7. REBUILD BEST MODEL
depth = best["max_depth"]

# handle None vs numeric depth
if pd.isna(depth):
    depth = None
else:
    depth = int(depth)

best_model = RandomForestClassifier(
    n_estimators=int(best["n_estimators"]),
    max_depth=depth,
    random_state=42
)

# retrain best model
best_model.fit(X_train, y_train)

# final predictions
y_pred = best_model.predict(X_test)


# 8. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# 9. FINAL METRICS
y_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("\nFinal Metrics:")
print("Accuracy:", best["test_acc"])
print("ROC-AUC:", roc_auc)