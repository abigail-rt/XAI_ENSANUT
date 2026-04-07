# -*- coding: utf-8 -*-

# ==========================
# IMPORTS
# ==========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score,
    precision_score, recall_score
)

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import shap


# ==========================
# LOAD DATA
# ==========================

df = pd.read_excel('../ENSANUT_engl.xlsx')

X = df.drop(['CSDS'], axis=1)
y = df['CSDS']


# ==========================
# SPLIT 70/30
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Original distribution TRAIN:")
print(y_train.value_counts())


# ==========================
# BALANCE STRATEGIES
# ==========================
balance_strategies = {
    "NO_BALANCE": None,
    "RUS": RandomUnderSampler(random_state=42),
    "ROS": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ENN": EditedNearestNeighbours(),
    "RUS_ROS": "RUS_ROS", 
    "SMOTE_ENN": SMOTEENN(random_state=42)
}


# ==========================
# MODELS
# ==========================
models = {
    "RF": RandomForestClassifier(random_state=42),
    "XGB": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    "LGBM": LGBMClassifier(random_state=42),
    "CAT": CatBoostClassifier(
        verbose=0,
        random_state=42
    )
}


# ==========================
# MAIN LOOP
# ==========================
for name, model in models.items():

    print("\n==============================")
    print("MODEL:", name)
    print("==============================")

    for strategy_name, sampler in balance_strategies.items():

        print("\nStrategy:", strategy_name)

        # -------- Balancing --------
        if strategy_name == "NO_BALANCE":
            X_tr, y_tr = X_train, y_train

        elif strategy_name == "RUS_ROS":
            ros = RandomOverSampler(random_state=42)
            rus = RandomUnderSampler(random_state=42)
            X_tmp, y_tmp = ros.fit_resample(X_train, y_train)
            X_tr, y_tr = rus.fit_resample(X_tmp, y_tmp)

        else:
            X_tr, y_tr = sampler.fit_resample(X_train, y_train)

        print("Distribution TRAIN:")
        print(pd.Series(y_tr).value_counts())

        # -------- Training --------
        model.fit(X_tr, y_tr)

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # -------- Metrics --------
        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='macro')
        precision_pos = precision_score(y_test, y_test_pred)
        recall_pos = recall_score(y_test, y_test_pred)

        print("Accuracy:", acc)
        print("F1 Macro:", f1)
        print("Precision (Pos):", precision_pos)
        print("Recall (Pos):", recall_pos)


print("\n===== EXPERIMENT FINISHED =====")


# ==========================
# XAI (SHAP)
# ==========================

for strategy_name, sampler in balance_strategies.items():

    print("\n==============================")
    print("Strategy:", strategy_name)
    print("==============================")

    # -------- Balancing --------
    if strategy_name == "NO_BALANCE":
        X_tr, y_tr = X_train, y_train

    elif strategy_name == "RUS_ROS":
        ros = RandomOverSampler(random_state=42)
        rus = RandomUnderSampler(random_state=42)
        X_tmp, y_tmp = ros.fit_resample(X_train, y_train)
        X_tr, y_tr = rus.fit_resample(X_tmp, y_tmp)

    else:
        X_tr, y_tr = sampler.fit_resample(X_train, y_train)

    # -------- Model --------
    model = LGBMClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------- Metrics --------
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Macro:", f1_score(y_test, y_pred, average='macro'))
    print("Precision (Pos):", precision_score(y_test, y_pred))
    print("Recall (Pos):", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("PR AUC:", average_precision_score(y_test, y_prob))

    # ==========================
    # SHAP
    # ==========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    plt.title(strategy_name)

    shap.summary_plot(shap_values, X_test, show=False)

    plt.savefig(f"shap_summary_{strategy_name}.png",
                dpi=300, bbox_inches="tight")

    plt.close()

print("\n===== XAI FINISHED =====")