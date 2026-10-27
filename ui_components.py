"""
UI Components and Theming Module for Netflix Subscriber Churn Intelligence.
----------------------------------------------------------------------------
Implements:
1. Custom CSS styling with charcoal/navy palette and restrained purple accents.
2. Prominent Synthetic Data Demo banners.
3. Metric KPI card generators.
4. Plotly charting functions with uniform dark aesthetic.
"""

from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants
BG_DARK = "#0C0F17"
CARD_BG = "#151B26"
BORDER_COLOR = "#222D3D"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_PURPLE_LIGHT = "#A78BFA"
ACCENT_VIOLET = "#7C3AED"
ACCENT_INDIGO = "#6366F1"
NETFLIX_RED = "#E50914"

TIER_COLORS = {
    "Low Risk": "#10B981",       # Emerald
    "Medium Risk": "#F59E0B",    # Amber
    "High Risk": "#F97316",      # Orange
    "Critical Risk": "#EF4444",  # Crimson
}


def inject_custom_css():
    """Injects bespoke dark navy & purple styling into Streamlit."""
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Background and container overrides */
    .stApp {{
        background-color: {BG_DARK};
        color: #E2E8F0;
    }}

    /* Global Synthetic Banner */
    .synthetic-banner {{
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.15) 0%, rgba(229, 9, 20, 0.15) 100%);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #E2E8F0;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }}

    .synthetic-badge {{
        background-color: #8B5CF6;
        color: #FFFFFF;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 10px;
    }}

    /* KPI Metric Cards */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .kpi-card:hover {{
        border-color: {ACCENT_PURPLE};
        transform: translateY(-2px);
    }}
    .kpi-title {{
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 6px;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }}
    .kpi-subtitle {{
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
    }}
    .kpi-delta-pos {{
        color: #10B981;
        font-weight: 600;
    }}
    .kpi-delta-neg {{
        color: #EF4444;
        font-weight: 600;
    }}

    /* Risk Tier Pills */
    .tier-pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .tier-low {{ background-color: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .tier-medium {{ background-color: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .tier-high {{ background-color: rgba(249, 115, 22, 0.18); color: #FB923C; border: 1px solid rgba(249, 115, 22, 0.4); }}
    .tier-critical {{ background-color: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }}

    /* Explanatory Callout Boxes */
    .callout-box {{
        background: #121722;
        border-left: 4px solid {ACCENT_PURPLE};
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
        color: #CBD5E1;
        font-size: 0.90rem;
    }}

    /* Sidebar Tweaks */
    section[data-testid="stSidebar"] {{
        background-color: #080B10;
        border-right: 1px solid {BORDER_COLOR};
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_synthetic_banner():
    """Renders prominent warning banner that all data is synthetic."""
    banner_html = """
    <div class="synthetic-banner">
        <div>
            <span class="synthetic-badge">Synthetic Demo Data</span>
            <strong>NOT REAL COMPANY DATA</strong> &mdash; Generated for fictional streaming churn modeling & retention simulation.
        </div>
        <div style="font-size: 0.80rem; opacity: 0.85;">
            Anchor: 2026 Simulation Cohort
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtitle: str = "", delta_type: str = "neutral"):
    """Renders styled dark navy KPI card."""
    delta_class = ""
    if delta_type == "positive":
        delta_class = "kpi-delta-pos"
    elif delta_type == "negative":
        delta_class = "kpi-delta-neg"

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle"><span class="{delta_class}">{subtitle}</span></div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# --- Plotly Charting Utilities ---

def apply_dark_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applies unified dark theme styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor="#1F2837",
        linecolor="#2D3748",
        zerolinecolor="#2D3748",
    )
    fig.update_yaxes(
        gridcolor="#1F2837",
        linecolor="#2D3748",
        zerolinecolor="#2D3748",
    )
    return fig


def plot_risk_tier_donut(df: pd.DataFrame) -> go.Figure:
    """Creates an interactive Donut chart of subscriber risk tiers."""
    tier_counts = df["risk_tier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]).fillna(0)
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=tier_counts.index,
                values=tier_counts.values,
                hole=0.62,
                marker=dict(colors=[TIER_COLORS[t] for t in tier_counts.index]),
                textinfo="percent+label",
                textposition="inside",
                insidetextorientation="radial",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Subscriber Risk Tier Distribution</b>", font=dict(size=15)),
        showlegend=False,
    )
    return apply_dark_theme(fig, height=340)


def plot_churn_by_plan(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing total subscribers and at-risk count by subscription tier."""
    plan_grp = df.groupby("plan_type").agg(
        Total=("subscriber_id", "count"),
        At_Risk=("churn_probability", lambda s: (s >= 0.50).sum()),
        Avg_Prob=("churn_probability", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Retained / Low Risk",
            x=plan_grp["plan_type"],
            y=plan_grp["Total"] - plan_grp["At_Risk"],
            marker_color="#4F46E5",
        )
    )
    fig.add_trace(
        go.Bar(
            name="At-Risk (P >= 0.50)",
            x=plan_grp["plan_type"],
            y=plan_grp["At_Risk"],
            marker_color=NETFLIX_RED,
        )
    )
    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Subscriber Portfolio by Plan Tier & At-Risk Status</b>", font=dict(size=14)),
        yaxis_title="Subscriber Count",
        xaxis_title="Plan Type",
    )
    return apply_dark_theme(fig, height=340)


def plot_inactivity_vs_viewing(df: pd.DataFrame) -> go.Figure:
    """Scatter plot highlighting days since last session vs viewing hours."""
    # Subsample 1200 rows if dataset is large to maintain crisp rendering
    plot_data = df.sample(n=min(len(df), 1200), random_state=42)

    fig = px.scatter(
        plot_data,
        x="days_since_last_session",
        y="viewing_hours_30d",
        color="churn_probability",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        size="monthly_fee",
        hover_data=["subscriber_id", "plan_type", "failed_payments_90d", "risk_tier"],
        labels={
            "days_since_last_session": "Days Inactive",
            "viewing_hours_30d": "30-Day Viewing Hours",
            "churn_probability": "Churn Risk",
            "monthly_fee": "Monthly Fee ($)",
        },
    )
    fig.update_layout(
        title=dict(text="<b>Inactivity vs Viewing Consumption (Colored by Churn Probability)</b>", font=dict(size=14)),
    )
    return apply_dark_theme(fig, height=380)


def plot_confusion_matrix(cm: List[List[int]]) -> go.Figure:
    """Heatmap of the classification confusion matrix."""
    z = cm
    x = ["Predicted Stay (0)", "Predicted Churn (1)"]
    y = ["Actual Stay (0)", "Actual Churn (1)"]

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"<b>{val:,}</b>",
                    showarrow=False,
                    font=dict(color="#FFFFFF", size=16),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#151B26"], [0.5, "#4338CA"], [1.0, "#8B5CF6"]],
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text="<b>Confusion Matrix (Held-out Test Cohort)</b>", font=dict(size=14)),
        annotations=annotations,
        yaxis=dict(autorange="reversed"),
    )
    return apply_dark_theme(fig, height=340)


def plot_roc_curves(rf_roc: Dict[str, Any], base_roc: Dict[str, Any], rf_auc: float, base_auc: float) -> go.Figure:
    """ROC Curve comparison: Primary Random Forest vs Baseline."""
    fig = go.Figure()
    
    # Baseline
    fig.add_trace(
        go.Scatter(
            x=base_roc["fpr"],
            y=base_roc["tpr"],
            mode="lines",
            name=f"Baseline Logistic (AUC = {base_auc:.3f})",
            line=dict(color="#94A3B8", dash="dash", width=2),
        )
    )
    # Random Forest
    fig.add_trace(
        go.Scatter(
            x=rf_roc["fpr"],
            y=rf_roc["tpr"],
            mode="lines",
            name=f"Random Forest (AUC = {rf_auc:.3f})",
            line=dict(color="#8B5CF6", width=3),
        )
    )
    # Diagonal line (random guess)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.50)",
            line=dict(color="#475569", dash="dot", width=1),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text="<b>Receiver Operating Characteristic (ROC Curve)</b>", font=dict(size=14)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return apply_dark_theme(fig, height=350)


def plot_feature_importance(feat_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top predictive features."""
    top_df = feat_df.head(top_n).sort_values(by="Importance", ascending=True)
    
    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],
            y=top_df["Feature"],
            orientation="h",
            marker=dict(
                color=top_df["Importance"],
                colorscale=[[0, "#4F46E5"], [1, "#8B5CF6"]],
            ),
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>Top {top_n} Churn Risk Predictor Signals (Feature Weights)</b>", font=dict(size=14)),
        xaxis_title="Relative Feature Importance",
        yaxis_title="Feature Name",
    )
    return apply_dark_theme(fig, height=350)


def plot_retention_action_roi(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing Gross Saved Revenue vs Total Cost across Action Categories."""
    action_grp = df.groupby("action_category").agg(
        Total_Cost=("intervention_cost", "sum"),
        Gross_Saved=("expected_gross_saved", "sum"),
        Net_Value=("expected_net_value", "sum"),
        Subscribers=("subscriber_id", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Intervention Cost ($)",
            x=action_grp["action_category"],
            y=action_grp["Total_Cost"],
            marker_color="#EF4444",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Expected Gross Saved Value ($)",
            x=action_grp["action_category"],
            y=action_grp["Gross_Saved"],
            marker_color="#10B981",
        )
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Retention Action Efficiency: Cost vs Expected Gross Saved</b>", font=dict(size=14)),
        xaxis_title="Action Category",
        yaxis_title="Amount ($ USD)",
    )
    return apply_dark_theme(fig, height=360)


def plot_subscriber_radar(subscriber_row: pd.Series, cohort_means: pd.Series) -> go.Figure:
    """Radar chart comparing an individual subscriber's risk metrics to cohort average."""
    metrics = [
        ("Inactivity (Days)", subscriber_row["days_since_last_session"], cohort_means["days_since_last_session"], 60),
        ("Viewing (Hrs)", subscriber_row["viewing_hours_30d"], cohort_means["viewing_hours_30d"], 100),
        ("Failed Payments", subscriber_row["failed_payments_90d"], cohort_means["failed_payments_90d"], 3),
        ("Support Tickets", subscriber_row["support_contacts_6m"], cohort_means["support_contacts_6m"], 5),
        ("Complaints", subscriber_row["complaints_logged"], cohort_means["complaints_logged"], 2),
        ("Completion Rate", subscriber_row["content_completion_rate"] * 100, cohort_means["content_completion_rate"] * 100, 100),
    ]

    categories = [m[0] for m in metrics]
    # Normalize each metric to 0-100 scale for clean radar presentation
    sub_vals = [min(100, max(0, (m[1] / m[3]) * 100)) for m in metrics]
    mean_vals = [min(100, max(0, (m[2] / m[3]) * 100)) for m in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=sub_vals + [sub_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Subscriber ({subscriber_row['subscriber_id']})",
            line=dict(color="#A78BFA", width=2),
            fillcolor="rgba(167, 139, 250, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=mean_vals + [mean_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Cohort Benchmark",
            line=dict(color="#64748B", dash="dash", width=1.5),
            fillcolor="rgba(100, 116, 139, 0.15)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1F2837", linecolor="#2D3748"),
            angularaxis=dict(gridcolor="#1F2837", linecolor="#2D3748"),
            bgcolor=CARD_BG,
        ),
        title=dict(text="<b>Subscriber Risk Signal Profile vs Cohort Average</b>", font=dict(size=14)),
        showlegend=True,
    )
    return apply_dark_theme(fig, height=360)
