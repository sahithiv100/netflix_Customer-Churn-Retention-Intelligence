"""
Netflix Subscriber Churn & Retention Intelligence Platform.
===========================================================
Interactive Streamlit Web Application.
DISCLAIMER: This application uses purely SYNTHETIC demo data.
It is an illustrative marketing analytics prototype and is NOT
affiliated with, scraped from, or endorsed by Netflix Inc.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Netflix Churn Intelligence | Marketing Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    inject_custom_css,
    render_synthetic_banner,
    render_kpi_card,
    plot_risk_tier_donut,
    plot_churn_by_plan,
    plot_inactivity_vs_viewing,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_retention_action_roi,
    plot_subscriber_radar,
    TIER_COLORS,
)

# Apply unified dark navy styling
inject_custom_css()

# ---------------------------------------------------------
# Session State Initialization & Data Caching
# ---------------------------------------------------------
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

if "sample_size" not in st.session_state:
    st.session_state.sample_size = 4000

if "horizon_months" not in st.session_state:
    st.session_state.horizon_months = 6

if "cost_multiplier" not in st.session_state:
    st.session_state.cost_multiplier = 1.0

if "efficacy_multiplier" not in st.session_state:
    st.session_state.efficacy_multiplier = 1.0

if "human_reviews" not in st.session_state:
    st.session_state.human_reviews = {}


@st.cache_data(show_spinner=False)
def load_and_score_data(sample_size: int, seed: int):
    """Generates synthetic data, trains ML model, and scores active cohort."""
    raw_df = generate_synthetic_subscribers(n_samples=sample_size, random_seed=seed)
    model_artifacts = train_churn_models(raw_df, random_seed=seed)
    scored_df = score_active_subscribers(raw_df, model_artifacts)
    return raw_df, model_artifacts, scored_df


# Load Data
with st.spinner("Initializing synthetic streaming dataset and training churn prediction models..."):
    raw_df, model_artifacts, scored_df = load_and_score_data(
        st.session_state.sample_size, st.session_state.random_seed
    )

# Apply retention engine calculations based on current scenario parameters
analyzed_df = assign_recommended_actions(
    scored_df,
    horizon_months=st.session_state.horizon_months,
    cost_multiplier=st.session_state.cost_multiplier,
    efficacy_multiplier=st.session_state.efficacy_multiplier,
)

# Filter active subscribers (the operational focus of marketing retention)
active_subs_df = analyzed_df[analyzed_df["cohort_status"] == "Active (To Score)"].copy()

# ---------------------------------------------------------
# Sidebar Navigation & Global Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <div style="background-color: #E50914; color: white; font-weight: 900; font-size: 1.3rem; padding: 4px 10px; border-radius: 4px;">N</div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em; color: #F8FAFC;">RETENTION AI</div>
                <div style="font-size: 0.70rem; color: #8B5CF6; font-weight: 600; text-transform: uppercase;">Marketing Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='background: #111827; border: 1px solid #374151; padding: 8px 12px; border-radius: 6px; font-size: 0.75rem; color: #9CA3AF; margin-bottom: 14px;'>"
        "⚠️ <strong>Synthetic Prototype</strong><br>Fictional Netflix demonstration. No proprietary data used."
        "</div>",
        unsafe_allow_html=True,
    )

    navigation_page = st.radio(
        "Navigation",
        [
            "Overview & Business Value",
            "Risk & Retention Dashboard",
            "Subscriber Risk Inspector",
            "Data Explorer & Dictionary",
            "How the Algorithm Works",
            "Model Performance & Audit",
            "Manager Questions & Scenarios",
            "Retention Campaign Hub",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Scenario Assumptions")

    new_horizon = st.slider(
        "Retention Horizon (Months)",
        min_value=3,
        max_value=12,
        value=st.session_state.horizon_months,
        step=3,
        help="Time window to compute subscriber lifetime value saved by intervention.",
    )
    if new_horizon != st.session_state.horizon_months:
        st.session_state.horizon_months = new_horizon
        st.rerun()

    new_cost_mult = st.slider(
        "Intervention Cost Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.cost_multiplier,
        step=0.1,
        help="Stress-test higher vendor/discount costs or operational economies of scale.",
    )
    if new_cost_mult != st.session_state.cost_multiplier:
        st.session_state.cost_multiplier = new_cost_mult
        st.rerun()

    new_eff_mult = st.slider(
        "Intervention Efficacy Multiplier",
        min_value=0.5,
        max_value=1.5,
        value=st.session_state.efficacy_multiplier,
        step=0.1,
        help="Adjust assumed conversion success of marketing interventions.",
    )
    if new_eff_mult != st.session_state.efficacy_multiplier:
        st.session_state.efficacy_multiplier = new_eff_mult
        st.rerun()

    if st.button("🔄 Reset Assumptions", use_container_width=True):
        st.session_state.horizon_months = 6
        st.session_state.cost_multiplier = 1.0
        st.session_state.efficacy_multiplier = 1.0
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎲 Synthetic Data Controls")

    col_seed1, col_seed2 = st.columns(2)
    with col_seed1:
        seed_input = st.number_input(
            "Seed", min_value=1, max_value=9999, value=st.session_state.random_seed, step=1
        )
    with col_seed2:
        size_input = st.selectbox(
            "Samples", [2000, 4000, 6000, 8000], index=[2000, 4000, 6000, 8000].index(st.session_state.sample_size)
        )

    if st.button("⚡ Regenerate Data", use_container_width=True):
        st.session_state.random_seed = int(seed_input)
        st.session_state.sample_size = int(size_input)
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "<div style='font-size: 0.72rem; color: #64748B; margin-top: 20px; text-align: center;'>"
        "Netflix Marketing Churn AI v2.4<br>Strictly Fictional Prototype"
        "</div>",
        unsafe_allow_html=True,
    )

# Display Global Top Banner
render_synthetic_banner()

# =========================================================
# PAGE 1: OVERVIEW & BUSINESS VALUE
# =========================================================
if navigation_page == "Overview & Business Value":
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.02em;">
                Subscriber Churn & Retention Intelligence
            </h1>
            <p style="font-size: 1.05rem; color: #94A3B8; max-width: 900px;">
                Predict fictional subscriber churn, pinpoint behavioral friction signals, and execute 
                cost-aware retention campaigns to maximize retained customer lifetime value.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Macro KPIs
    total_active = len(active_subs_df)
    critical_subs = len(active_subs_df[active_subs_df["risk_tier"] == "Critical Risk"])
    high_subs = len(active_subs_df[active_subs_df["risk_tier"] == "High Risk"])
    at_risk_subs = critical_subs + high_subs
    total_rev_at_risk = active_subs_df["revenue_at_risk"].sum()
    total_net_saved = active_subs_df["expected_net_value"].sum()
    avg_churn_risk = active_subs_df["churn_probability"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Active Scoring Cohort",
            f"{total_active:,}",
            f"{len(raw_df):,} total synthetic records",
            "neutral",
        )
    with c2:
        render_kpi_card(
            "At-Risk Subscribers",
            f"{at_risk_subs:,}",
            f"{(at_risk_subs/total_active)*100:.1f}% in High/Critical Tier",
            "negative",
        )
    with c3:
        render_kpi_card(
            f"{st.session_state.horizon_months}-Mo Revenue at Risk",
            f"${total_rev_at_risk:,.0f}",
            f"Avg ${total_rev_at_risk/total_active:.1f} per active sub",
            "negative",
        )
    with c4:
        render_kpi_card(
            "Expected Net Saved Value",
            f"${total_net_saved:,.0f}",
            f"ROI: {(total_net_saved / max(1, active_subs_df['intervention_cost'].sum())) * 100:.0f}% on spend",
            "positive",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Narrative Columns: The Business Problem & Strategy
    col_left, col_right = st.columns([1.1, 0.9])
    with col_left:
        st.markdown("### 🎯 The Streaming Retention Challenge")
        st.markdown(
            """
            In high-volume subscription streaming platforms like Netflix, subscriber acquisition costs 
            (CAC) continue to escalate while content competition intensifies. Blanket retention promotions 
            (such as untargeted price discounts) destroy unit economics by giving discounts to satisfied users 
            who would have stayed anyway, while failing to address root-cause friction for dissatisfied users.

            This decision-support prototype solves this dilemma by marrying **predictive machine learning** 
            with **cost-aware intervention economics**:
            - **Prevent Data Leakage:** Separates historical subscriber journeys from current live cohorts.
            - **Calibrate Risk Tiers:** Distinguishes healthy engaged accounts from High and Critical flight risks.
            - **Attribute Actionable Signals:** Differentiates between involuntary payment friction and organic content disengagement.
            - **Prescribe Cost-Aware Interventions:** Allocates marketing dollars only where the expected net retained value strictly exceeds intervention cost.
            """
        )

    with col_right:
        st.markdown("### 💡 Four Tailored Marketing Interventions")
        st.markdown(
            """
            <div style="background: #141923; border: 1px solid #232A3B; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <strong style="color: #F87171;">1. Smart Payment Rescue (Dunning Help)</strong><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Friction-free SMS & 1-click tokenized update with a 7-day streaming grace period. Tackles 
                    involuntary churn before bank cancellation.
                </span>
            </div>
            <div style="background: #141923; border: 1px solid #232A3B; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <strong style="color: #A78BFA;">2. AI-Powered Catalog Discovery</strong><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Personalized push/email highlighting new episodes in the user's primary genre and top 10 trends 
                    to overcome browsing paralysis.
                </span>
            </div>
            <div style="background: #141923; border: 1px solid #232A3B; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <strong style="color: #60A5FA;">3. Household Onboarding & Device Setup</strong><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Incentivizes creating family sub-profiles and pairing smart TVs or mobile offline downloads, 
                    deepening switching barriers.
                </span>
            </div>
            <div style="background: #141923; border: 1px solid #232A3B; border-radius: 8px; padding: 14px 18px;">
                <strong style="color: #FBBF24;">4. Plan Right-Sizing & Loyalty Credit</strong><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Proactive option to switch to Standard-with-Ads ($6.99/mo) or apply a temporary loyalty credit 
                    to preserve subscriber lifetime value.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Portfolio Snapshot")
    snap1, snap2 = st.columns([1, 1])
    with snap1:
        st.plotly_chart(plot_risk_tier_donut(active_subs_df), use_container_width=True)
    with snap2:
        st.plotly_chart(plot_churn_by_plan(active_subs_df), use_container_width=True)

# =========================================================
# PAGE 2: RISK & RETENTION DASHBOARD
# =========================================================
elif navigation_page == "Risk & Retention Dashboard":
    st.markdown("## 📊 Risk & Retention Dashboard")
    st.markdown("Comprehensive view of active subscribers, churn vulnerability, and retention campaign efficiency.")

    # KPI Row
    total_active = len(active_subs_df)
    critical_count = len(active_subs_df[active_subs_df["risk_tier"] == "Critical Risk"])
    high_count = len(active_subs_df[active_subs_df["risk_tier"] == "High Risk"])
    tot_cost = active_subs_df["intervention_cost"].sum()
    tot_gross = active_subs_df["expected_gross_saved"].sum()
    tot_net = active_subs_df["expected_net_value"].sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Critical Risk Subs", f"{critical_count:,}", f"{(critical_count/total_active)*100:.1f}% of cohort", "negative")
    with k2:
        render_kpi_card("High Risk Subs", f"{high_count:,}", f"{(high_count/total_active)*100:.1f}% of cohort", "negative")
    with k3:
        render_kpi_card("Required Budget", f"${tot_cost:,.0f}", f"Avg ${tot_cost/total_active:.2f} per subscriber", "neutral")
    with k4:
        render_kpi_card("Net Retained Value", f"${tot_net:,.0f}", f"Net ROI: {(tot_net/tot_cost)*100:.1f}%", "positive")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(plot_risk_tier_donut(active_subs_df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(plot_retention_action_roi(active_subs_df), use_container_width=True)

    st.markdown("### 🔍 Inactivity vs Consumption vs Churn Probability")
    st.plotly_chart(plot_inactivity_vs_viewing(active_subs_df), use_container_width=True)

    # Top At-Risk Subscribers Preview
    st.markdown("### 🚨 Highest Priority At-Risk Subscribers")
    st.markdown("Ranked by expected net retained revenue saved through automated interventions.")
    
    top_at_risk = active_subs_df.sort_values(by="expected_net_value", ascending=False).head(15)[
        [
            "subscriber_id",
            "plan_type",
            "monthly_fee",
            "tenure_months",
            "days_since_last_session",
            "failed_payments_90d",
            "churn_probability",
            "risk_tier",
            "recommended_action",
            "intervention_cost",
            "expected_net_value",
        ]
    ]
    st.dataframe(
        top_at_risk.style.format(
            {
                "monthly_fee": "${:.2f}",
                "churn_probability": "{:.1%}",
                "intervention_cost": "${:.2f}",
                "expected_net_value": "${:.2f}",
            }
        ),
        use_container_width=True,
    )

# =========================================================
# PAGE 3: SUBSCRIBER RISK INSPECTOR
# =========================================================
elif navigation_page == "Subscriber Risk Inspector":
    st.markdown("## 🔍 Subscriber Risk Inspector")
    st.markdown("Deep-dive into individual synthetic subscriber profiles, inspect attribution signals, and review retention actions.")

    # Search and Filter Controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            ["All Tiers", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"],
            index=1,
        )
    with f_col2:
        filtered_sub_pool = active_subs_df
        if tier_filter != "All Tiers":
            filtered_sub_pool = active_subs_df[active_subs_df["risk_tier"] == tier_filter]

        sample_ids = filtered_sub_pool["subscriber_id"].head(100).tolist()
        if not sample_ids:
            st.warning("No subscribers found matching the selected filter.")
            selected_sub_id = active_subs_df["subscriber_id"].iloc[0]
        else:
            selected_sub_id = st.selectbox("Select Subscriber ID to Inspect", sample_ids)

    # Retrieve subscriber details
    sub_row = active_subs_df[active_subs_df["subscriber_id"] == selected_sub_id].iloc[0]
    cohort_means = active_subs_df.mean(numeric_only=True)

    st.markdown("---")
    
    # Customer Profile Cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_kpi_card("Plan & Monthly Fee", f"{sub_row['plan_type']}", f"${sub_row['monthly_fee']:.2f} / month", "neutral")
    with p2:
        render_kpi_card("Tenure & Cohort", f"{sub_row['tenure_months']} Months", f"Active Live Cohort", "neutral")
    with p3:
        risk_color = "negative" if sub_row["churn_probability"] >= 0.50 else "positive"
        render_kpi_card("Model Churn Probability", f"{sub_row['churn_probability'] * 100:.1f}%", f"{sub_row['risk_tier']}", risk_color)
    with p4:
        render_kpi_card("Expected Net Value Saved", f"${sub_row['expected_net_value']:.2f}", f"Intervention Cost: ${sub_row['intervention_cost']:.2f}", "positive")

    # Radar Chart & Behavioral Indicators
    col_radar, col_behavior = st.columns([1.1, 0.9])
    with col_radar:
        st.plotly_chart(plot_subscriber_radar(sub_row, cohort_means), use_container_width=True)

    with col_behavior:
        st.markdown("### 📋 Behavioral Health Metrics")
        st.markdown(
            f"""
            - **Recent Inactivity**: `{sub_row['days_since_last_session']} days` since last watch session (Cohort avg: `{cohort_means['days_since_last_session']:.1f}d`)
            - **30-Day Stream Volume**: `{sub_row['viewing_hours_30d']} hours` across `{sub_row['active_days_30d']} active days`
            - **Content Completion**: `{sub_row['content_completion_rate']*100:.0f}%` of started titles finished
            - **Payment Health**: `{sub_row['failed_payments_90d']} failed billing events` in last 90 days
            - **Support Friction**: `{sub_row['support_contacts_6m']} contacts`, `{sub_row['complaints_logged']} escalated complaints`
            - **Household Adoption**: `{sub_row['profile_count']} profiles` on `{sub_row['device_count']} devices` (Favorite Genre: `{sub_row['primary_genre']}`)
            """
        )

        st.markdown("### ⚠️ Primary Risk Attribution")
        st.markdown(
            f"""
            <div class="callout-box">
                <strong>Signal Diagnosis:</strong> {sub_row['primary_risk_driver']}<br>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">
                    <em>Note: Attribution signals represent correlated statistical risk factors and do not establish deterministic causation.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action Prescription Card & Human Sign-off
    st.markdown("### 🎯 Recommended Retention Action")
    act_col1, act_col2 = st.columns([1.2, 0.8])
    with act_col1:
        st.markdown(
            f"""
            <div style="background: #151B26; border: 1px solid #7C3AED; border-radius: 8px; padding: 18px 22px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.82rem; text-transform: uppercase; font-weight: 700; color: #A78BFA;">
                        {sub_row['action_category']} Intervention
                    </span>
                    <span style="background: rgba(124, 58, 237, 0.2); color: #C4B5FD; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        Priority Rank #{sub_row['priority_rank']}
                    </span>
                </div>
                <h3 style="margin-top: 0; color: #F8FAFC;">{sub_row['recommended_action']}</h3>
                <p style="color: #CBD5E1; font-size: 0.90rem;">{DEFAULT_INTERVENTIONS[sub_row['action_category']]['description']}</p>
                <div style="display: flex; gap: 24px; margin-top: 14px; border-top: 1px solid #232A3B; padding-top: 12px; font-size: 0.85rem;">
                    <div><strong>Channel:</strong> {DEFAULT_INTERVENTIONS[sub_row['action_category']]['channel']}</div>
                    <div><strong>Unit Cost:</strong> ${sub_row['intervention_cost']:.2f}</div>
                    <div><strong>Expected Efficacy:</strong> {sub_row['intervention_efficacy']*100:.1f}%</div>
                    <div><strong>Expected ROI:</strong> {sub_row['expected_roi_pct']:.0f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with act_col2:
        st.markdown("### 👤 Human Review & Execution Sign-Off")
        
        # Check current review state
        current_status = st.session_state.human_reviews.get(selected_sub_id, "Pending Review")
        
        status_color = "#F59E0B" if current_status == "Pending Review" else "#10B981" if "Approved" in current_status else "#EF4444"
        st.markdown(
            f"<div style='margin-bottom: 12px; font-size: 0.90rem;'>Current Status: "
            f"<strong style='color: {status_color};'>{current_status}</strong></div>",
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Approve Action", key=f"app_{selected_sub_id}", use_container_width=True):
                st.session_state.human_reviews[selected_sub_id] = "Approved by Marketing Lead"
                st.success("Intervention campaign approved for CRM automation.")
                st.rerun()
        with b2:
            if st.button("🛑 Escalate / Hold", key=f"esc_{selected_sub_id}", use_container_width=True):
                st.session_state.human_reviews[selected_sub_id] = "Escalated for VIP Concierge Call"
                st.warning("Subscriber routed to manual customer support review.")
                st.rerun()

        if st.button("🚫 Flag as Do Not Contact", key=f"dnc_{selected_sub_id}", use_container_width=True):
            st.session_state.human_reviews[selected_sub_id] = "Excluded (Do Not Contact)"
            st.info("Subscriber excluded from promotional outreach.")
            st.rerun()

# =========================================================
# PAGE 4: DATA EXPLORER & DICTIONARY
# =========================================================
elif navigation_page == "Data Explorer & Dictionary":
    st.markdown("## 📁 Data Explorer & Synthetic Data Dictionary")
    st.markdown("Inspect synthetic subscriber records, apply custom cohort filters, and download dataset exports.")

    tab_explore, tab_dict = st.tabs(["📊 Interactive Data Explorer", "📖 Data Dictionary & Schema"])

    with tab_explore:
        st.markdown("### Filter Subscribers")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        with exp_col1:
            sel_cohort = st.multiselect(
                "Cohort Status",
                options=analyzed_df["cohort_status"].unique(),
                default=["Active (To Score)"],
            )
        with exp_col2:
            sel_plans = st.multiselect(
                "Plan Types",
                options=analyzed_df["plan_type"].unique(),
                default=list(analyzed_df["plan_type"].unique()),
            )
        with exp_col3:
            sel_tiers = st.multiselect(
                "Risk Tiers",
                options=analyzed_df["risk_tier"].unique(),
                default=list(analyzed_df["risk_tier"].unique()),
            )
        with exp_col4:
            max_inactivity = st.slider("Min Days Inactive", 0, 60, 0)

        # Apply Filters
        filtered_df = analyzed_df[
            (analyzed_df["cohort_status"].isin(sel_cohort))
            & (analyzed_df["plan_type"].isin(sel_plans))
            & (analyzed_df["risk_tier"].isin(sel_tiers))
            & (analyzed_df["days_since_last_session"] >= max_inactivity)
        ]

        st.markdown(f"**Showing {len(filtered_df):,} matching subscribers** (out of {len(analyzed_df):,} total synthetic records)")
        st.dataframe(filtered_df.head(200), use_container_width=True)

        # Download Buttons
        st.markdown("### 📥 Download Exports")
        d1, d2 = st.columns(2)
        with d1:
            csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Filtered Records (CSV)",
                data=csv_filtered,
                file_name="netflix_synthetic_filtered_subscribers.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            at_risk_export = analyzed_df[analyzed_df["risk_tier"].isin(["High Risk", "Critical Risk"])].sort_values(
                by="expected_net_value", ascending=False
            )
            csv_at_risk = at_risk_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download At-Risk Action List (CSV)",
                data=csv_at_risk,
                file_name="netflix_synthetic_at_risk_priority_list.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_dict:
        st.markdown("### 📖 Synthetic Data Dictionary")
        st.markdown(
            "Every column in this prototype represents an engineered synthetic metric mimicking streaming service "
            "dynamics. Bounds and business contexts are documented below:"
        )
        dict_df = get_data_dictionary()
        st.dataframe(dict_df, use_container_width=True, height=500)

# =========================================================
# PAGE 5: HOW THE ALGORITHM WORKS
# =========================================================
elif navigation_page == "How the Algorithm Works":
    st.markdown("## ⚙️ How the Algorithm Works")
    st.markdown("An end-to-end audit of data simulation, leakage prevention, machine learning, and decision economics.")

    st.markdown(
        """
        <div style="background: #141923; border: 1px solid #232A3B; border-radius: 8px; padding: 22px; margin-bottom: 25px;">
            <h3 style="margin-top: 0; color: #A78BFA;">🔄 Connected 6-Stage Intelligence Pipeline</h3>
            <div style="display: flex; flex-direction: column; gap: 14px; font-size: 0.92rem; color: #E2E8F0; margin-top: 15px;">
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">1</div>
                    <div>
                        <strong>Synthetic Generation & Temporal Cohort Partitioning:</strong><br>
                        Fictional subscriber records are generated with realistic joint distributions. Crucially, subscribers are partitioned into <em>Historical (Train/Test)</em> and <em>Active (To Score)</em> cohorts based on signup timeline, preventing future lookahead leakage.
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">2</div>
                    <div>
                        <strong>Feature Engineering & Multi-Channel Preprocessing:</strong><br>
                        Behavioral telemetry (viewing hours, active days, recency), customer service friction (contacts, complaints), payment health (failed dunning), and subscription tier variables are preprocessed and one-hot encoded.
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">3</div>
                    <div>
                        <strong>Supervised Model Training & Cross-Model Auditing:</strong><br>
                        A transparent baseline (Standard Scaled Logistic Regression) and an ensemble classifier (Random Forest with 120 estimators and depth constraints) are trained exclusively on historical data and benchmarked on held-out test splits.
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">4</div>
                    <div>
                        <strong>Calibrated Risk Scoring & Tier Segmentation:</strong><br>
                        Active subscribers receive continuous churn probabilities <code>P(churn) &in; [0, 1]</code> and are categorized into Low (&lt;20%), Medium (20-49%), High (50-74%), or Critical (&ge;75%) risk tiers.
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">5</div>
                    <div>
                        <strong>Non-Causal Signal Attribution:</strong><br>
                        The algorithm identifies leading correlated friction points for each subscriber (e.g. repeated dunning failure vs catalog disengagement) without falsely asserting causal claims.
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="background: #7C3AED; color: white; border-radius: 50%; min-width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700;">6</div>
                    <div>
                        <strong>Cost-Aware Retention Optimization & Ranking:</strong><br>
                        Interventions are matched to risk drivers. The financial equation ranks subscribers by Expected Net Retained Value:
                        <br>
                        <code>Net Retained Value = (Monthly Fee &times; Horizon) &times; P(churn) &times; Efficacy &minus; Intervention Cost</code>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧮 Mathematical & Economic Foundations")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            r"""
            #### 1. Churn Probability & Calibrated Risk
            The ensemble classifier outputs an estimated probability $P(\text{churn} \mid X)$. 
            A subscriber's revenue exposed to loss over an evaluation horizon $H$ (in months) is:
            $$\text{Revenue at Risk} = (\text{Monthly Fee} \times H) \times P(\text{churn})$$
            
            Subscribers are segmented into four risk bands:
            * **Low Risk**: $P(\text{churn}) < 0.20$
            * **Medium Risk**: $0.20 \le P(\text{churn}) < 0.50$
            * **High Risk**: $0.50 \le P(\text{churn}) < 0.75$
            * **Critical Risk**: $P(\text{churn}) \ge 0.75$
            """
        )
    with m_col2:
        st.markdown(
            r"""
            #### 2. Net Retained Value & Optimization
            Let $C_i$ be the cost of intervention $i$ and $E_i \in (0, 1)$ be the empirical retention efficacy rate:
            $$\text{Gross Saved Value} = \text{Revenue at Risk} \times E_i$$
            $$\text{Net Retained Value} = \text{Gross Saved Value} - C_i$$
            $$\text{Expected ROI} = \frac{\text{Net Retained Value}}{C_i} \times 100\%$$

            An intervention is dispatched **only if Net Retained Value > 0**, ensuring every campaign dollar deployed is accretive.
            """
        )

# =========================================================
# PAGE 6: MODEL PERFORMANCE & AUDIT
# =========================================================
elif navigation_page == "Model Performance & Audit":
    st.markdown("## 📈 Model Performance & Audit")
    st.markdown("Transparent audit of classification metrics on held-out test data, ROC curves, and feature contributions.")

    rf_metrics = model_artifacts["rf_metrics"]
    base_metrics = model_artifacts["base_metrics"]

    # Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi_card("Accuracy", f"{rf_metrics['accuracy']:.1%}", f"Baseline: {base_metrics['accuracy']:.1%}", "positive")
    with m2:
        render_kpi_card("Precision", f"{rf_metrics['precision']:.1%}", f"Baseline: {base_metrics['precision']:.1%}", "positive")
    with m3:
        render_kpi_card("Recall", f"{rf_metrics['recall']:.1%}", f"Baseline: {base_metrics['recall']:.1%}", "positive")
    with m4:
        render_kpi_card("F1-Score", f"{rf_metrics['f1']:.3f}", f"Baseline: {base_metrics['f1']:.3f}", "positive")
    with m5:
        render_kpi_card("ROC-AUC", f"{rf_metrics['roc_auc']:.3f}", f"Baseline: {base_metrics['roc_auc']:.3f}", "positive")

    st.markdown("<br>", unsafe_allow_html=True)
    c_roc, c_cm = st.columns([1.1, 0.9])
    with c_roc:
        st.plotly_chart(
            plot_roc_curves(
                rf_metrics["roc_curve"],
                base_metrics["roc_curve"],
                rf_metrics["roc_auc"],
                base_metrics["roc_auc"],
            ),
            use_container_width=True,
        )
    with c_cm:
        st.plotly_chart(plot_confusion_matrix(rf_metrics["confusion_matrix"]), use_container_width=True)

    st.markdown("### 📊 Predictor Signal Importance")
    st.plotly_chart(plot_feature_importance(model_artifacts["feature_importance"], top_n=10), use_container_width=True)

    # Disclosures & Limitations
    st.markdown("### ⚠️ Model Limitations & Ethical Disclosures")
    st.markdown(
        """
        <div style="background: #141923; border-left: 4px solid #E50914; padding: 14px 18px; border-radius: 0 8px 8px 0; font-size: 0.88rem; color: #CBD5E1;">
            <strong>1. Correlation vs Causation:</strong> High feature importance (e.g. days since last watch or billing failures) demonstrates statistical correlation with subscriber cancellation, not immutable causation. Interventions address symptoms and friction points to restore affinity.<br><br>
            <strong>2. Transparent Synthetic Training:</strong> All training labels are synthetically simulated. If evaluated in production, continuous retraining with real observational dunning logs and A/B test feedback loops would be mandated.<br><br>
            <strong>3. Human Review for High-Cost Actions:</strong> Direct concession offers (like multi-month discounts) require marketing manager approval to safeguard gross margins.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 7: MANAGER QUESTIONS & SCENARIOS
# =========================================================
elif navigation_page == "Manager Questions & Scenarios":
    st.markdown("## 💡 Manager Questions & Scenario Simulator")
    st.markdown("Explore strategic marketing decisions and simulate live organizational scenarios.")

    mq_tabs = st.tabs([
        "Q1: Cohort Revenue Risk vs Cost to Save",
        "Q2: Payment Gateway Failure Surge Simulation",
        "Q3: Constrained Budget Optimization (Knapsack)",
    ])

    # Manager Question 1
    with mq_tabs[0]:
        st.markdown("### Manager Question 1")
        st.markdown(
            "***'Which subscriber cohort generates the highest revenue at risk, and is it cheaper to save them or lower-tier subscribers?'***"
        )
        q1_results = solve_manager_question_1(active_subs_df)
        st.markdown(q1_results["insight"])

        st.dataframe(
            q1_results["summary_table"].style.format(
                {
                    "Total Subscribers": "{:,}",
                    "At-Risk Subscribers": "{:,}",
                    "At-Risk Rate (%)": "{:.1f}%",
                    "Total Revenue at Risk ($)": "${:,.2f}",
                    "Total Intervention Cost ($)": "${:,.2f}",
                    "Expected Net Saved Value ($)": "${:,.2f}",
                    "Average Cost / Sub ($)": "${:.2f}",
                    "Portfolio ROI (%)": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        st.plotly_chart(plot_churn_by_plan(active_subs_df), use_container_width=True)

    # Manager Question 2
    with mq_tabs[1]:
        st.markdown("### Manager Question 2")
        st.markdown(
            "***'If payment gateway failure rate spikes by X% (e.g., bank mandate changes, card expirations, or processing outage), how many subscribers slip into Critical risk and what is the automated recovery ROI?'***"
        )

        spike_val = st.slider("Simulated Payment Failure Surge (%)", 0, 100, 25, 5)
        q2_res = solve_manager_question_2(active_subs_df, payment_failure_spike_pct=spike_val)

        col_q2_1, col_q2_2, col_q2_3, col_q2_4 = st.columns(4)
        with col_q2_1:
            render_kpi_card(
                "Critical Risk Subs",
                f"{q2_res['simulated_critical']:,}",
                f"+{q2_res['critical_subscribers_added']} from baseline",
                "negative",
            )
        with col_q2_2:
            render_kpi_card(
                "Affected Subscribers",
                f"{q2_res['affected_subscribers']:,}",
                f"Surge Factor: +{spike_val}%",
                "neutral",
            )
        with col_q2_3:
            render_kpi_card(
                "Automated Rescue Cost",
                f"${q2_res['rescue_cost']:,.0f}",
                "@ $2.50 / SMS rescue flow",
                "neutral",
            )
        with col_q2_4:
            render_kpi_card(
                "Net Recovered Revenue",
                f"${q2_res['net_rescue_benefit']:,.0f}",
                f"ROI: {q2_res['rescue_roi_pct']:.0f}%",
                "positive",
            )

        st.markdown(
            f"""
            <div class="callout-box">
                <strong>Strategic Takeaway:</strong> A <strong>{spike_val}% surge</strong> in payment gateway friction increases 
                critical-tier flight risks by <strong>{q2_res['critical_subscribers_added']} subscribers</strong>. 
                Deploying automated <em>Smart Payment Rescue</em> recaptures an estimated 
                <strong>${q2_res['recovered_revenue']:,.2f}</strong> in streaming revenue against a modest 
                <strong>${q2_res['rescue_cost']:,.2f}</strong> SMS/dunning operational spend, achieving an outstanding 
                <strong>{q2_res['rescue_roi_pct']:.0f}% Net ROI</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Manager Question 3
    with mq_tabs[2]:
        st.markdown("### Manager Question 3")
        st.markdown(
            "***'What is the optimal intervention budget allocation under constrained marketing spend?'***"
        )
        st.markdown(
            "Solves a greedy knapsack optimization by ranking subscribers according to **Expected Net Retained Value per Dollar Spent**."
        )

        budget_slider = st.slider("Marketing Retention Budget Cap ($)", 1000, 30000, 8000, 1000)
        q3_res = solve_manager_question_3(active_subs_df, budget_cap=budget_slider)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            render_kpi_card("Treated Subscribers", f"{q3_res['subscribers_treated']:,}", f"{q3_res['subscribers_unfunded']:,} unfunded", "neutral")
        with b2:
            render_kpi_card("Total Budget Spent", f"${q3_res['total_spent']:,.0f}", f"${q3_res['remaining_budget']:,.0f} unspent", "neutral")
        with b3:
            render_kpi_card("Total Gross Saved", f"${q3_res['total_gross_saved']:,.0f}", "Revenue preserved", "positive")
        with b4:
            render_kpi_card("Net Retained Value", f"${q3_res['total_net_saved']:,.0f}", f"ROI: {q3_res['portfolio_roi_pct']:.0f}%", "positive")

        st.markdown("#### Optimal Interventions Deployed Under Budget Cap")
        action_counts = pd.DataFrame(
            list(q3_res["action_breakdown"].items()), columns=["Action Category", "Subscribers Targeted"]
        )
        st.dataframe(action_counts, use_container_width=True)

# =========================================================
# PAGE 8: RETENTION CAMPAIGN HUB
# =========================================================
elif navigation_page == "Retention Campaign Hub":
    st.markdown("## 🎯 Retention Campaign Hub")
    st.markdown("Operational dispatch list for marketing campaigns across SMS, dynamic email, in-app push, and retention concessions.")

    # Top Filters
    c_f1, c_f2 = st.columns([1, 1])
    with c_f1:
        sel_campaign_action = st.selectbox(
            "Select Campaign Channel / Action",
            ["All Interventions", "Payment Help", "Discovery", "Onboarding", "Plan Review"],
        )
    with c_f2:
        only_approved = st.checkbox("Show Only Human-Approved High-Impact Actions", value=False)

    campaign_df = active_subs_df.copy()
    if sel_campaign_action != "All Interventions":
        campaign_df = campaign_df[campaign_df["action_category"] == sel_campaign_action]

    if only_approved:
        approved_ids = [k for k, v in st.session_state.human_reviews.items() if "Approved" in v]
        campaign_df = campaign_df[campaign_df["subscriber_id"].isin(approved_ids)]

    st.markdown(f"**Target Audience: {len(campaign_df):,} Subscribers**")
    
    # Priority Table
    st.dataframe(
        campaign_df[
            [
                "priority_rank",
                "subscriber_id",
                "plan_type",
                "churn_probability",
                "risk_tier",
                "recommended_action",
                "intervention_cost",
                "expected_gross_saved",
                "expected_net_value",
                "primary_risk_driver",
            ]
        ]
        .sort_values(by="priority_rank")
        .head(100)
        .style.format(
            {
                "churn_probability": "{:.1%}",
                "intervention_cost": "${:.2f}",
                "expected_gross_saved": "${:.2f}",
                "expected_net_value": "${:.2f}",
            }
        ),
        use_container_width=True,
    )

    # Export Button
    st.markdown("### 📤 Campaign Export")
    campaign_csv = campaign_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export Campaign Target List for Marketing Automation (CSV)",
        data=campaign_csv,
        file_name=f"netflix_retention_campaign_{sel_campaign_action.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
