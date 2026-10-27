import sys
import os

# Add directory to sys.path
sys.path.insert(0, r"C:\Users\DELL\.gemini\antigravity\scratch\netflix_churn_intelligence")

print("--- Testing Imports ---")
from synthetic_data import generate_synthetic_subscribers, get_data_dictionary
from model_engine import train_churn_models, score_active_subscribers
from retention_engine import (
    assign_recommended_actions,
    solve_manager_question_1,
    solve_manager_question_2,
    solve_manager_question_3,
    DEFAULT_INTERVENTIONS,
)
from ui_components import (
    plot_risk_tier_donut,
    plot_churn_by_plan,
    plot_inactivity_vs_viewing,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_retention_action_roi,
    plot_subscriber_radar,
)
print("All modules imported successfully!")

print("\n--- Testing Synthetic Data Generation ---")
df = generate_synthetic_subscribers(n_samples=1000, random_seed=42)
print(f"Generated DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Cohort status counts:\n{df['cohort_status'].value_counts()}")
dd = get_data_dictionary()
print(f"Data dictionary records: {len(dd)}")

print("\n--- Testing Model Training ---")
model_artifacts = train_churn_models(df, random_seed=42)
print(f"RF Accuracy: {model_artifacts['rf_metrics']['accuracy']:.3f}, AUC: {model_artifacts['rf_metrics']['roc_auc']:.3f}")
print(f"Baseline Accuracy: {model_artifacts['base_metrics']['accuracy']:.3f}, AUC: {model_artifacts['base_metrics']['roc_auc']:.3f}")
print(f"Top 3 Features:\n{model_artifacts['feature_importance'].head(3)}")

print("\n--- Testing Active Scoring ---")
scored_df = score_active_subscribers(df, model_artifacts)
print(f"Risk tier counts:\n{scored_df['risk_tier'].value_counts()}")

print("\n--- Testing Retention Engine ---")
analyzed_df = assign_recommended_actions(scored_df, horizon_months=6, cost_multiplier=1.0, efficacy_multiplier=1.0)
print(f"Action category counts:\n{analyzed_df['action_category'].value_counts()}")
print(f"Mean Expected Net Value: ${analyzed_df['expected_net_value'].mean():.2f}")

print("\n--- Testing Manager Questions ---")
active_df = analyzed_df[analyzed_df['cohort_status'] == 'Active (To Score)']
q1 = solve_manager_question_1(active_df)
print(f"Q1 summary shape: {q1['summary_table'].shape}")
q2 = solve_manager_question_2(active_df, payment_failure_spike_pct=25.0)
print(f"Q2 critical delta: {q2['critical_subscribers_added']}, net benefit: ${q2['net_rescue_benefit']}")
q3 = solve_manager_question_3(active_df, budget_cap=5000.0)
print(f"Q3 treated subs: {q3['subscribers_treated']}, total spent: ${q3['total_spent']}")

print("\n--- Testing UI Plots ---")
fig1 = plot_risk_tier_donut(active_df)
fig2 = plot_churn_by_plan(active_df)
fig3 = plot_inactivity_vs_viewing(active_df)
fig4 = plot_confusion_matrix(model_artifacts['rf_metrics']['confusion_matrix'])
fig5 = plot_roc_curves(
    model_artifacts['rf_metrics']['roc_curve'],
    model_artifacts['base_metrics']['roc_curve'],
    model_artifacts['rf_metrics']['roc_auc'],
    model_artifacts['base_metrics']['roc_auc'],
)
fig6 = plot_feature_importance(model_artifacts['feature_importance'])
fig7 = plot_retention_action_roi(active_df)
sub_row = active_df.iloc[0]
cohort_means = active_df.mean(numeric_only=True)
fig8 = plot_subscriber_radar(sub_row, cohort_means)
print("All 8 Plotly figures generated successfully!")

print("\n--- All Tests Passed Successfully! ---")
