# Netflix Subscriber Churn & Retention Intelligence Platform

> **DISCLAIMER:**  
> **SYNTHETIC DEMO DATA — NOT REAL COMPANY DATA.**  
> This platform is an illustrative data science and marketing prototype created strictly for educational, demonstration, and portfolio purposes. It does not use, scrape, or imply access to real Netflix data, customer records, proprietary algorithms, or internal billing infrastructure.

---

## 📌 Executive Summary

Subscription streaming platforms face escalating customer acquisition costs (CAC) alongside catalog competition. Blanket retention tactics (such as untargeted discounts) destroy subscription margins by rewarding loyal subscribers who would have renewed anyway, while failing to resolve operational friction for dissatisfied customers.

This application marries **supervised machine learning** with **cost-aware microeconomics** to predict fictional subscriber churn, attribute behavioral and payment friction signals without false causal claims, and prescribe optimal interventions that maximize net retained customer lifetime value.

---

## 🚀 Key Features

1. **Synthetic Data Engine (`synthetic_data.py`)**:
   - Generates realistic streaming subscription datasets including tenure, plan tiers (Standard with Ads, Standard, Premium), monthly fees, 30-day viewing volume, daily active days, inactivity recency, title completion rates, failed payment dunning events, support tickets, and complaints.
   - Built-in temporal cohort partitioning (`Historical` vs `Active`) to eliminate lookahead data leakage.
   - Reproducible random seed control and interactive on-the-fly regeneration.

2. **Tailored Machine Learning Engine (`model_engine.py`)**:
   - Compares a transparent baseline (Logistic Regression) against a production ensemble (Random Forest Classifier).
   - Computes empirical metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
   - Scores active subscribers with calibrated churn probabilities $P(\text{churn}) \in [0, 1]$.
   - Segments accounts into four risk tiers: Low (<20%), Medium (20–49%), High (50–74%), and Critical (≥75%).
   - Attributes individual risk drivers without claiming causal certainty.

3. **Cost-Aware Retention Engine (`retention_engine.py`)**:
   - Maps risk signals to 4 cost-aware interventions:
     - **Payment Help**: Automated dunning rescue & 7-day grace period.
     - **Discovery**: AI-powered personalized catalog & top 10 re-engagement.
     - **Onboarding**: Family sub-profiles & multi-device household setup.
     - **Plan Review**: Proactive right-sizing to Standard-with-Ads or temporary loyalty credits.
   - Calculates **Expected Net Retained Value**:
     $$\text{Net Value} = (\text{Monthly Fee} \times \text{Horizon}) \times P(\text{churn}) \times \text{Efficacy} - \text{Intervention Cost}$$
   - Ranks interventions strictly by expected net profit, discarding non-accretive actions.

4. **Strategic Manager Questions & Scenarios**:
   - **Q1**: Cohort Revenue Risk vs Cost to Save (comparing plan elasticity and net ROI).
   - **Q2**: Payment Gateway Failure Surge Simulation (simulating bank mandate failures and automated rescue ROI).
   - **Q3**: Constrained Budget Optimization (greedy knapsack solver for optimal marketing dollar allocation).

5. **Interactive UI & Human-in-the-Loop (`ui_components.py`, `app.py`)**:
   - Dark charcoal/navy interface with restrained purple/violet accents.
   - High-impact human review sign-off module for high-cost retention concessions.
   - Dynamic data explorer with multi-column filtering and instant CSV export buttons.

---

## 🛠️ Windows Installation & Running Instructions

### Step 1: Open PowerShell or Command Prompt
Navigate to the project directory:
```powershell
cd C:\Users\DELL\.gemini\antigravity\scratch\netflix_churn_intelligence
```

### Step 2: (Recommended) Create & Activate Virtual Environment
Using your preferred Python 3.10+ installation:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Or if using Anaconda / Miniconda):*
```powershell
conda activate base
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will automatically open your default web browser at `http://localhost:8501`.

---

## 📂 File Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Main Streamlit application with multi-page navigation, session management, and UI logic. |
| `synthetic_data.py` | Synthetic dataset generator, realistic streaming probability formulas, and data dictionary. |
| `model_engine.py` | Data partitioning, baseline & Random Forest training, metric audits, and active scoring. |
| `retention_engine.py` | Business rules, intervention assignment, net retained value formulas, and manager solvers. |
| `ui_components.py` | Custom dark navy CSS styling, synthetic warning banners, KPI cards, and Plotly charts. |
| `requirements.txt` | Minimal, locked Python dependencies (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`). |
| `README.md` | Comprehensive system documentation, usage guide, and audit methodology. |

---

## 🧪 Testing & Validation

The codebase has undergone verification tests:
1. **Module Import & Dependencies**: Verified that `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `plotly` load cleanly without warnings.
2. **Synthetic Data Consistency**: Validated that distributions, correlated engagement variables, and temporal partitions maintain realistic bounds.
3. **Model Convergence**: Verified that the Random Forest model and Logistic Regression baseline train and score within < 2 seconds, producing valid non-null confusion matrices and ROC-AUC curves.
4. **Interactive Controls & Downloads**: Verified that CSV downloads, filters, scenario sliders, seed controls, and human-in-the-loop review state persist seamlessly.

---

## ⚖️ Limitations & Ethical Disclosures
- **Synthetic Demonstration**: All customer metrics, churn labels, and response curves are synthetic. Real-world deployment requires continuous calibration with live payment gateways and A/B test holdout cohorts.
- **Correlation vs Causation**: Feature importance scores represent statistical correlations rather than direct causal drivers. Interventions are hypotheses to be tested empirically.
- **Human Oversight**: High-cost concession interventions require marketing manager review before dispatch to safeguard subscription gross margins.
