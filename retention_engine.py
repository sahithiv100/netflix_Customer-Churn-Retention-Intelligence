"""
Retention Engine Module for Netflix Subscriber Churn Intelligence.
------------------------------------------------------------------
Implements:
1. Business logic mapping risk signals to 4 cost-aware interventions:
   - Payment Help (Smart Payment Rescue & Grace Period)
   - Discovery (Personalized Catalog Watchlist Push)
   - Onboarding (Family & Multi-Device Household Setup)
   - Plan Review (Tier Right-Sizing & Loyalty Credit)
2. Expected Retained Value & Net ROI calculations.
3. Ranking interventions by expected net retained value.
4. Editable scenario assumptions calculator.
5. Analytical solvers for the three key manager questions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Default Intervention Parameters
DEFAULT_INTERVENTIONS = {
    "Payment Help": {
        "action_name": "Smart Dunning & SMS Payment Rescue",
        "description": "Friction-free 1-click card update SMS/email with 7-day streaming grace period.",
        "base_cost": 2.50,
        "base_efficacy": 0.48,  # 48% chance of saving subscriber
        "channel": "Automated SMS / In-App Modal",
        "human_review_required": False,
    },
    "Discovery": {
        "action_name": "Personalized Catalog Re-Engagement",
        "description": "Targeted push + email highlighting trending titles in subscriber's top genre.",
        "base_cost": 0.85,
        "base_efficacy": 0.24,  # 24% chance of saving subscriber
        "channel": "Dynamic Email + App Push",
        "human_review_required": False,
    },
    "Onboarding": {
        "action_name": "Family & Multi-Device Household Guide",
        "description": "Guided profile setup, mobile offline download walkthrough, and TV pairing.",
        "base_cost": 1.40,
        "base_efficacy": 0.20,  # 20% chance of saving subscriber
        "channel": "In-App Feature Spotlight",
        "human_review_required": False,
    },
    "Plan Review": {
        "action_name": "Plan Right-Sizing & Loyalty Concession",
        "description": "Proactive switch to Standard-with-Ads or 3-month $5 loyalty credit on current plan.",
        "base_cost": 16.00,  # Promo credit + billing adjustment cost
        "base_efficacy": 0.58,  # 58% chance of saving subscriber
        "channel": "VIP Retention Email / In-App Offer",
        "human_review_required": True,  # Higher cost requires policy or human sign-off
    },
}


def assign_recommended_actions(
    df: pd.DataFrame,
    horizon_months: int = 6,
    cost_multiplier: float = 1.0,
    efficacy_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Assigns the most suitable retention action to each active subscriber based on risk signals.
    Computes Expected Gross Saved Value, Intervention Cost, Net Retained Value, and ROI.
    """
    res_df = df.copy()

    assigned_actions: List[str] = []
    action_names: List[str] = []
    costs: List[float] = []
    efficacies: List[float] = []
    human_reviews: List[bool] = []

    for _, row in res_df.iterrows():
        churn_p = row.get("churn_probability", 0.15)
        
        # Action Assignment Priority Hierarchy:
        # Priority 1: Billing/Payment failure (Involuntary churn risk)
        if row["failed_payments_90d"] >= 1:
            category = "Payment Help"
        # Priority 2: High complaints, expensive plan, or very high risk with tenure >= 6m
        elif (row["complaints_logged"] >= 1 or row["monthly_fee"] >= 15.49) and churn_p >= 0.55:
            category = "Plan Review"
        # Priority 3: Low household penetration (single profile, single device, early tenure)
        elif row["profile_count"] == 1 and row["device_count"] <= 2 and row["tenure_months"] <= 6:
            category = "Onboarding"
        # Priority 4: Disengagement / Inactivity -> Content discovery push
        else:
            category = "Discovery"

        cfg = DEFAULT_INTERVENTIONS[category]
        unit_cost = cfg["base_cost"] * cost_multiplier
        eff = np.clip(cfg["base_efficacy"] * efficacy_multiplier, 0.05, 0.95)

        assigned_actions.append(category)
        action_names.append(cfg["action_name"])
        costs.append(round(unit_cost, 2))
        efficacies.append(round(eff, 3))
        human_reviews.append(cfg["human_review_required"] and churn_p >= 0.60)

    res_df["action_category"] = assigned_actions
    res_df["recommended_action"] = action_names
    res_df["intervention_cost"] = costs
    res_df["intervention_efficacy"] = efficacies
    res_df["human_review_required"] = human_reviews

    # Financial Computations:
    # 1. Customer Horizon Value = monthly_fee * horizon_months
    # 2. Revenue at Risk = Customer Horizon Value * churn_probability
    # 3. Expected Gross Saved Value = Revenue at Risk * intervention_efficacy
    # 4. Net Retained Value = Expected Gross Saved Value - intervention_cost
    # 5. Expected ROI % = (Net Retained Value / intervention_cost) * 100
    res_df["horizon_months"] = horizon_months
    res_df["customer_horizon_revenue"] = np.round(res_df["monthly_fee"] * horizon_months, 2)
    res_df["revenue_at_risk"] = np.round(res_df["customer_horizon_revenue"] * res_df["churn_probability"], 2)
    res_df["expected_gross_saved"] = np.round(res_df["revenue_at_risk"] * res_df["intervention_efficacy"], 2)
    res_df["expected_net_value"] = np.round(res_df["expected_gross_saved"] - res_df["intervention_cost"], 2)
    res_df["expected_roi_pct"] = np.where(
        res_df["intervention_cost"] > 0,
        np.round((res_df["expected_net_value"] / res_df["intervention_cost"]) * 100, 1),
        0.0,
    )

    # Priority Rank: Descending order of expected net retained value
    res_df["priority_rank"] = res_df["expected_net_value"].rank(ascending=False, method="min").astype(int)

    return res_df


def solve_manager_question_1(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Manager Question 1:
    'Which subscriber cohort generates the highest revenue at risk, and is it cheaper to save them
    or lower-tier subscribers?'
    """
    cohort_stats = []
    for plan, group in df.groupby("plan_type"):
        total_subs = len(group)
        at_risk_subs = len(group[group["churn_probability"] >= 0.50])
        total_rev_at_risk = group["revenue_at_risk"].sum()
        total_cost = group["intervention_cost"].sum()
        total_gross_saved = group["expected_gross_saved"].sum()
        total_net_value = group["expected_net_value"].sum()
        avg_cost_per_sub = total_cost / total_subs if total_subs > 0 else 0
        roi = (total_net_value / total_cost * 100) if total_cost > 0 else 0

        cohort_stats.append(
            {
                "Plan": plan,
                "Total Subscribers": total_subs,
                "At-Risk Subscribers": at_risk_subs,
                "At-Risk Rate (%)": round((at_risk_subs / total_subs) * 100, 1) if total_subs else 0,
                "Total Revenue at Risk ($)": round(total_rev_at_risk, 2),
                "Total Intervention Cost ($)": round(total_cost, 2),
                "Expected Net Saved Value ($)": round(total_net_value, 2),
                "Average Cost / Sub ($)": round(avg_cost_per_sub, 2),
                "Portfolio ROI (%)": round(roi, 1),
            }
        )

    summary_df = pd.DataFrame(cohort_stats).sort_values(by="Total Revenue at Risk ($)", ascending=False)
    highest_rev_cohort = summary_df.iloc[0]["Plan"]

    # Comparison insight
    finding = (
        f"**Key Finding**: The **{highest_rev_cohort}** tier generates the highest absolute Revenue at Risk "
        f"(${summary_df.iloc[0]['Total Revenue at Risk ($)']:,.0f}) due to higher monthly fees. However, "
        f"Standard with Ads and Standard subscribers often deliver higher percentage ROI on digital nudges "
        f"(email/push) because intervention delivery costs are virtually identical, whereas Premium subscribers "
        f"frequently require costlier concessions (loyalty credits) to achieve retention."
    )

    return {"summary_table": summary_df, "insight": finding}


def solve_manager_question_2(
    df: pd.DataFrame, payment_failure_spike_pct: float = 25.0
) -> Dict[str, Any]:
    """
    Manager Question 2:
    'If payment gateway failure rate spikes by X% (e.g., bank mandate changes or expired cards),
    how many subscribers slip into Critical risk and what is the automated recovery ROI?'
    """
    simulated_df = df.copy()
    
    # Baseline counts
    baseline_critical = int(len(simulated_df[simulated_df["risk_tier"] == "Critical Risk"]))
    baseline_high_critical = int(len(simulated_df[simulated_df["churn_probability"] >= 0.50]))

    # Calculate additional subscribers impacted by payment gateway failure surge
    base_failures_count = int((simulated_df["failed_payments_90d"] > 0).sum())
    n_additional = int(np.round(base_failures_count * (payment_failure_spike_pct / 100.0)))
    if n_additional == 0 and payment_failure_spike_pct > 0:
        n_additional = max(1, int(len(simulated_df) * 0.03 * (payment_failure_spike_pct / 25.0)))

    # Select candidates to experience payment gateway bounces (skewed toward shorter tenures & non-premium)
    eligible_mask = simulated_df["failed_payments_90d"] < 3
    candidate_indices = simulated_df[eligible_mask].sort_values(by=["churn_probability"], ascending=False).index

    affected_indices = candidate_indices[:n_additional]
    simulated_df["sim_churn_probability"] = simulated_df["churn_probability"]
    
    # Each payment failure escalates churn probability significantly (+0.30)
    simulated_df.loc[affected_indices, "sim_churn_probability"] = np.clip(
        simulated_df.loc[affected_indices, "churn_probability"] + 0.32, 0.01, 0.99
    )

    sim_critical = int(np.sum(simulated_df["sim_churn_probability"] >= 0.75))
    critical_delta = max(0, sim_critical - baseline_critical)

    # Automated Smart Payment Rescue impact on the newly affected pool
    # Smart dunning cost: $2.50, efficacy: 48%
    n_affected = len(affected_indices)
    rescue_cost = n_affected * 2.50
    avg_horizon_rev = simulated_df.loc[affected_indices, "customer_horizon_revenue"].mean() if n_affected > 0 else simulated_df["customer_horizon_revenue"].mean()
    recovered_revenue = n_affected * 0.48 * avg_horizon_rev
    net_rescue_benefit = recovered_revenue - rescue_cost
    rescue_roi = (net_rescue_benefit / rescue_cost * 100) if rescue_cost > 0 else 0

    return {
        "spike_pct": payment_failure_spike_pct,
        "baseline_critical": baseline_critical,
        "simulated_critical": sim_critical,
        "critical_subscribers_added": critical_delta,
        "affected_subscribers": n_affected,
        "rescue_cost": round(rescue_cost, 2),
        "recovered_revenue": round(recovered_revenue, 2),
        "net_rescue_benefit": round(net_rescue_benefit, 2),
        "rescue_roi_pct": round(rescue_roi, 1),
    }


def solve_manager_question_3(df: pd.DataFrame, budget_cap: float = 10000.0) -> Dict[str, Any]:
    """
    Manager Question 3:
    'What is the optimal intervention budget allocation under constrained marketing spend?'
    Solves greedy knapsack by prioritizing highest Expected Net Retained Value per dollar of cost.
    """
    # Filter subscribers where an intervention yields positive net value
    candidates = df[df["expected_net_value"] > 0].copy()
    
    # Efficiency ratio: Expected Net Value per Dollar of Intervention Cost
    candidates["value_per_cost"] = candidates["expected_net_value"] / np.maximum(candidates["intervention_cost"], 0.1)
    candidates = candidates.sort_values(by="value_per_cost", ascending=False).reset_index(drop=True)

    # Cumulative cost greedy selection
    candidates["cumulative_spend"] = candidates["intervention_cost"].cumsum()
    selected = candidates[candidates["cumulative_spend"] <= budget_cap].copy()
    unfunded = candidates[candidates["cumulative_spend"] > budget_cap].copy()

    total_spent = selected["intervention_cost"].sum()
    total_gross_saved = selected["expected_gross_saved"].sum()
    total_net_saved = selected["expected_net_value"].sum()
    subscribers_treated = len(selected)
    subscribers_unfunded = len(unfunded)
    unfunded_rev_at_risk = unfunded["revenue_at_risk"].sum() if len(unfunded) > 0 else 0.0

    action_breakdown = selected["action_category"].value_counts().to_dict()

    return {
        "budget_cap": budget_cap,
        "total_spent": round(total_spent, 2),
        "remaining_budget": round(budget_cap - total_spent, 2),
        "subscribers_treated": subscribers_treated,
        "subscribers_unfunded": subscribers_unfunded,
        "total_gross_saved": round(total_gross_saved, 2),
        "total_net_saved": round(total_net_saved, 2),
        "portfolio_roi_pct": round((total_net_saved / total_spent * 100), 1) if total_spent > 0 else 0,
        "unfunded_revenue_at_risk": round(unfunded_rev_at_risk, 2),
        "action_breakdown": action_breakdown,
        "selected_sample": selected.head(10),
    }
