# Factors Associated with Depression in Mexico: An XAI Comparative Analysis
Code related to the article Factors Associated with Depression in Mexico: An XAI Comparative Analysis of Tree-Based Classifiers Under Imbalanced Data.

Dataset

The dataset can be found at: https://doi.org/10.5281/zenodo.19458275

Requirements

Install the required libraries using:

pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost imbalanced-learn shap openpyxl

Methodology

The following resampling techniques were evaluated:

Random Undersampling (RUS)
Random Oversampling (ROS)
SMOTE
Edited Nearest Neighbours (ENN)
RUS + ROS
SMOTE + ENN

The models used include:

Random Forest
XGBoost
LightGBM
CatBoost

Performance was evaluated using standard classification metrics (Accuracy, F1-score, Precision, Recall, ROC-AUC, PR-AUC).

Reference

For more details on SHAP, see:
https://shap.readthedocs.io/en/latest/
