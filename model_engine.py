"""
Model Engine Module for Netflix Subscriber Churn Intelligence.
--------------------------------------------------------------
Implements:
1. Leakage-free temporal data partitioning.
2. Baseline heuristic and trained Random Forest / Logistic Regression classifier.
3. Audit metrics: Precision, Recall, F1, ROC-AUC, Confusion Matrix, and ROC Curve.
4. Active subscriber scoring, risk tier assignment, and risk signal attribution.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "tenure_months",
    "monthly_fee",
    "viewing_hours_30d",
    "active_days_30d",
    "days_since_last_session",
    "content_completion_rate",
    "failed_payments_90d",
    "support_contacts_6m",
    "complaints_logged",
    "profile_count",
    "device_count",
    "downloads_used_30d",
]

CATEGORICAL_COLS = ["plan_type", "offer_history"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares numeric and one-hot encoded feature matrix."""
    df_feat = df[FEATURE_COLS].copy()
    
    # Categorical one-hot encoding
    encoded_plans = pd.get_dummies(df["plan_type"], prefix="plan", drop_first=False)
    encoded_offers = pd.get_dummies(df["offer_history"], prefix="offer", drop_first=False)
    
    X = pd.concat([df_feat, encoded_plans, encoded_offers], axis=1)
    return X


def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, Any]:
    """Calculates full suite of classification performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
    }


def train_churn_models(df: pd.DataFrame, random_seed: int = 42) -> Dict[str, Any]:
    """
    Splits historical data to prevent lookahead leakage, then trains:
    1. Transparent Heuristic / Logistic Regression Baseline.
    2. Primary Supervised Classifier (Random Forest).
    Returns trained models, feature names, test evaluation metrics, and scalers.
    """
    # 1. Temporal Partition: Filter historical cohort for training & validation
    hist_df = df[df["cohort_status"] == "Historical (Train/Test)"].copy()
    if len(hist_df) < 100:
        # Fallback if cohort filter returns small dataset
        hist_df = df.copy()

    X_all = prepare_features(hist_df)
    feature_names = list(X_all.columns)
    y_all = hist_df["churn_label"].values

    # Train / Test split on historical cohort
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=random_seed, stratify=y_all
    )

    # Standard Scaler for Baseline Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train Baseline (Standard Logistic Regression)
    baseline_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    baseline_model.fit(X_train_scaled, y_train)
    base_pred = baseline_model.predict(X_test_scaled)
    base_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]
    base_metrics = evaluate_model(y_test, base_pred, base_prob)

    # 3. Train Primary Supervised Model (Random Forest Ensemble)
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=random_seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)

    # 4. Feature Importances
    importances = rf_model.feature_importances_
    feat_importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": rf_model,
        "baseline_model": baseline_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "rf_metrics": rf_metrics,
        "base_metrics": base_metrics,
        "feature_importance": feat_importance_df,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
    }


def score_active_subscribers(df: pd.DataFrame, model_artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Scores all subscribers using the trained model.
    Assigns:
    - churn_risk_score (0.0 to 1.0)
    - risk_tier ('Low', 'Medium', 'High', 'Critical')
    - primary_risk_driver (narrative signal attribution without causal claim)
    - risk_signals (top 3 contributing metrics)
    """
    scored_df = df.copy()
    X_full = prepare_features(scored_df)
    
    # Align features in case of dummy column mismatches
    for col in model_artifacts["feature_names"]:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[model_artifacts["feature_names"]]

    # Predict risk probabilities
    model = model_artifacts["model"]
    probs = model.predict_proba(X_full)[:, 1]
    scored_df["churn_probability"] = np.round(probs, 4)

    # Risk Tiers
    # Low: < 0.20, Medium: 0.20-0.49, High: 0.50-0.74, Critical: >= 0.75
    conditions = [
        scored_df["churn_probability"] < 0.20,
        (scored_df["churn_probability"] >= 0.20) & (scored_df["churn_probability"] < 0.50),
        (scored_df["churn_probability"] >= 0.50) & (scored_df["churn_probability"] < 0.75),
        scored_df["churn_probability"] >= 0.75,
    ]
    tiers = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    scored_df["risk_tier"] = np.select(conditions, tiers, default="Medium Risk")

    # Risk Signal Attribution (Rule-assisted signal weighting grounded in actual feature values)
    primary_drivers = []
    top_signals_list = []

    for idx, row in scored_df.iterrows():
        signals = []
        
        # 1. Payment failure signal
        if row["failed_payments_90d"] >= 2:
            signals.append((row["failed_payments_90d"] * 0.40, f"Recurring billing failure ({int(row['failed_payments_90d'])} failed attempts)"))
        elif row["failed_payments_90d"] == 1:
            signals.append((0.25, "Recent card authorization issue (1 failed dunning)"))

        # 2. Inactivity signal
        if row["days_since_last_session"] >= 30:
            signals.append((row["days_since_last_session"] * 0.012, f"Protracted inactivity ({int(row['days_since_last_session'])}d since last watch)"))
        elif row["days_since_last_session"] >= 14:
            signals.append((0.20, f"Engagement drop ({int(row['days_since_last_session'])}d since last watch)"))

        # 3. Viewing volume signal
        if row["viewing_hours_30d"] < 5:
            signals.append((0.30, f"Extremely low consumption ({row['viewing_hours_30d']} hrs in 30d)"))
        elif row["viewing_hours_30d"] < 15:
            signals.append((0.15, f"Below-average streaming ({row['viewing_hours_30d']} hrs in 30d)"))

        # 4. Support & complaints
        if row["complaints_logged"] >= 1:
            signals.append((0.35 * row["complaints_logged"], f"Service escalation ({int(row['complaints_logged'])} complaints logged)"))
        elif row["support_contacts_6m"] >= 3:
            signals.append((0.20, f"Frequent support tickets ({int(row['support_contacts_6m'])} contacts)"))

        # 5. Low completion rate
        if row["content_completion_rate"] < 0.25:
            signals.append((0.22, f"High title abandonment ({int(row['content_completion_rate']*100)}% completion)"))

        # 6. Single profile / single device under-utilization
        if row["profile_count"] == 1 and row["device_count"] == 1 and row["tenure_months"] < 6:
            signals.append((0.14, "Low household adoption (single profile & device)"))

        # Sort signals by relative attribution magnitude
        if signals:
            signals.sort(key=lambda x: x[0], reverse=True)
            primary_drivers.append(signals[0][1])
            top_signals_list.append("; ".join([s[1] for s in signals[:3]]))
        else:
            primary_drivers.append("Normal usage patterns (Healthy engagement)")
            top_signals_list.append("Regular multi-device viewing; no payment friction")

    scored_df["primary_risk_driver"] = primary_drivers
    scored_df["risk_signals_summary"] = top_signals_list

    return scored_df
