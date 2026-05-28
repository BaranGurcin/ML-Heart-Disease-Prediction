# Heart Disease Prediction Using Machine Learning

This project applies multiple machine learning algorithms to the **Heart Disease Dataset** to compare their performance in predicting heart disease.
The project includes:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting with Missing Value Imputation

The goal is to evaluate different models, feature selection techniques, stability analysis methods, and missing value handling strategies for binary classification problems in healthcare data.

---

# Project Structure

```bash
├── heart.csv
├── Logistic_Reggesion.py
├── Random_Forest.py
├── SVM.py
├── GradientBoosting_Imputation.py
└── README.md
```

---

# Dataset

The project uses the **Heart Disease Dataset**.

Expected target column:

```python
target
```

* `0` → No heart disease
* `1` → Heart disease detected

---

# Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-ml.git
cd heart-disease-ml
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

# Models Implemented

## 1. Logistic Regression

File:

```bash
Logistic_Reggesion.py
```

Features included:

* Standard scaling
* Baseline model evaluation
* Stability analysis
* Mutual Information feature selection
* Recursive Feature Elimination (RFE)
* Performance comparison

Metrics:

* Accuracy
* F1-score
* ROC-AUC

---

## 2. Random Forest

File:

```bash
Random_Forest.py
```

Features included:

* Hyperparameter tuning
* Different tree depths
* Different estimator counts
* Best model selection
* Confusion matrix analysis

Metrics:

* Accuracy
* ROC-AUC

---

## 3. Support Vector Machine (SVM)

File:

```bash
SVM.py
```

Models tested:

* Linear Kernel
* RBF Kernel

Features included:

* Cross-validation
* Stability analysis
* Overfitting analysis
* Train/Test comparison
* Confusion matrices

Metrics:

* Accuracy
* F1-score
* ROC-AUC

---

## 4. Gradient Boosting with Missing Value Imputation

File:

```bash
GradientBoosting_Imputation.py
```

Features included:

* Artificial missing value generation
* Mean Imputation
* KNN Imputation
* Gradient Boosting classification
* Imputation method comparison

Metrics:

* Accuracy
* F1-score
* ROC-AUC
* Confusion Matrix

---

# How to Run

Run each script individually:

## Logistic Regression

```bash
python Logistic_Reggesion.py
```

## Random Forest

```bash
python Random_Forest.py
```

## SVM

```bash
python SVM.py
```

## Gradient Boosting

```bash
python GradientBoosting_Imputation.py
```

---

# Evaluation Metrics

The project evaluates models using:

* Accuracy
* F1-score
* ROC-AUC
* Confusion Matrix
* Cross-validation statistics
* Stability analysis

---

# Machine Learning Concepts Covered

This project demonstrates several important machine learning concepts:

* Data preprocessing
* Feature scaling
* Missing value handling
* Feature selection
* Hyperparameter tuning
* Cross-validation
* Stability analysis
* Overfitting detection
* Model comparison

---

# Example Output

Example evaluation table:

| Model               | Accuracy | F1-score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 0.85     | 0.86     | 0.91    |
| Random Forest       | 0.89     | 0.88     | 0.93    |
| SVM RBF             | 0.90     | 0.89     | 0.94    |

---

# Future Improvements

Possible future improvements include:

* XGBoost implementation
* Deep Learning models
* Feature engineering
* SHAP explainability
* GridSearchCV optimization
* Deployment with Flask or FastAPI

---

Developed as a machine learning project for heart disease prediction and model comparison using Python and Scikit-learn.
