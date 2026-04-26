# Churn ROI

> Predicting customer churn for a telecom provider and quantifying the revenue saved by targeted retention interventions.

**Status:** Pipeline complete. 

---

## The Business Problem

Customer churn is expensive. For a subscription business, losing a customer means losing not just this month's revenue but the entire remaining customer lifetime. But retention interventions also cost money (discounts), support calls, and contract incentives eat into margin. The question is *"at what churn probability does intervention pay for itself, and which intervention pays for itself fastest?"*

This project answers those questions using the Telco Customer Churn Kaggle dataset, a standard benchmark of ~7,000 telecom customers with subscription details and churn labels.

## Key Questions

1. Which customers are most likely to churn, and what drives their risk?
2. How much monthly recurring revenue (MRR) is at risk?
3. At what predicted churn probability should the business intervene, given the cost and success rate of interventions?
4. Among possible retention strategies (discounts, contract upgrades, premium support), which has the best ROI?

## Why This Is More Than a Churn Classifier
The main goal is to support a retention budget decision: which customers should receive an intervention, and which retention strategy has the highest expected return?

The model output is converted into an economic decision rule:

intervene iff P(churn) × retention_success_rate × CLV > intervention_cost.

Because the optimal threshold depends on business assumptions, the project includes cost-sensitive threshold optimization, ROI-ranked retention strategies, and sensitivity analysis across intervention cost and success rate. The reported business lift is measured against a do-nothing baseline and can be reproduced with `scripts/compute_headline_metrics.py`.

The dataset is a public benchmark, so the novelty is not the raw churn prediction task. The value is the end-to-end decision framework: model comparison, MRR-at-risk analysis, SQL cohort views, dashboard-ready exports, and explicit assumptions that can be replaced with real company data.

## Headline Results

> Numbers below are produced by `scripts/compute_headline_metrics.py` running the full pipeline against the Kaggle dataset. Reproduce with `python scripts/compute_headline_metrics.py`.

| Metric | Value | Source |
|---|---|---|
| Customer base | 7,032 (after dropping 11 tenure=0 rows) | `data_loader.load_clean()` |
| Overall churn rate | **26.6%** | `01_eda.ipynb` |
| Logistic regression test AUC | **0.835** | `02_baseline_model.ipynb` |
| XGBoost test AUC | **0.830** | `03_xgboost.ipynb` |
| Total MRR | **$455,661** | `01_eda.ipynb` § 5 |
| 12-month MRR from actual churners | **$1,669,570** | `01_eda.ipynb` § 5 |
| Optimal intervention threshold τ* | **0.10** | `04_cost_analysis.ipynb` |
| Lift vs. do-nothing baseline (test set) | **+$144,379** over 12 months | `04_cost_analysis.ipynb` |
| Top strategy by **net value** | Combined discount + addon (ROI 1.42×) | `06_retention_strategies.ipynb` |
| Top strategy by **ROI** | Free security addon, 6mo (ROI 2.56×) | `06_retention_strategies.ipynb` |

**Note on AUC:** logistic regression edges out XGBoost by 0.005 here — meaningfully tied at this dataset size. With ~7k rows and well-separated bucketed features, the linear model captures most of the signal. 

**Note on the τ\* = 0.10:** the optimal threshold falls below the modeling default of 0.5 because the average CLV ($65/mo × 12 = $780) is large relative to the $50 intervention cost — even modestly-likely churners are worth targeting. The notebook's sensitivity heatmap shows τ\* climbing above 0.5 only when intervention cost exceeds $150 or success rate falls below 15%.

**Note on the negative `net_value` numbers in the strategy table:** `net_value = saved_revenue − intervention_costs − unrecovered_churn_losses`, so it is total profit, not delta vs. baseline. The headline number to quote is **lift vs. do-nothing**

---

## Methodology

### Data
- **Source:** [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers, 21 features
- **Target:** Binary `Churn` label
- **Loading:** via `kagglehub` (see `data/README.md`); the loader handles the known `TotalCharges` blank-row for tenure=0 customers.

### Pipeline
1. **EDA** (`01_eda.ipynb`) — churn rate, feature-target relationships, MRR at risk.
2. **Baseline model** (`02_baseline_model.ipynb`) — logistic regression to handle the 73/27 split.
3. **Tree model** (`03_xgboost.ipynb`) — XGBoost vs LR on the same split.
4. **Cost-sensitive analysis** (`04_cost_analysis.ipynb`) — sweep thresholds, find the τ that maximizes net value, run a 5×5 sensitivity heatmap 
5. **SQL feature engineering** (`sql/01_feature_engineering.sql`, `sql/02_at_risk_cohorts.sql`) — analyst-facing views: bucketed features, priority cohorts, KPI aggregates.
6. **Dashboard export** (`05_dashboard_export.ipynb`) — orchestrates everything: out-of-fold CV predictions for every customer, SQLite load, runs the SQL files, exports Tableau-ready CSVs.
7. **Strategy ranking** (`06_retention_strategies.ipynb`) — each retention play (discount / addon / contract upgrade / premium support / combined) becomes a (cost, success_rate) tuple. Each strategy gets its own optimal threshold; we compare on both net value and ROI, with a ±30% sensitivity check on the success-rate assumption.

### Key Assumptions

These are placeholders for the user to override with real ops data. They live in `src/cost_analysis.CostAssumptions`:

| Parameter | Default | Rationale |
|---|---|---|
| Intervention cost | $50 / targeted customer | Approximate cost of a retention call plus a small discount |
| Intervention success rate | 30% | Industry-typical for outbound retention; should be measured via A/B test |
| CLV horizon | 12 months of MRR | Simple horizon-based CLV; decoupled from threshold sensitivity |

The decision rule reduces to: **intervene iff `P(churn) × success_rate × CLV > intervention_cost`**.

### Class imbalance and metric choice

Churn is ~27% positive. Accuracy is misleading at this rate. Reported metrics lead with **AUC** and **recall on the churn class**; the cost-sensitive analysis takes precedence over any metric for the deployment decision.

---

## Project Structure

```
churn-predictor/
├── data/
│   ├── README.md                  # Kagglehub download + schema
│   └── (gitignored: CSVs, SQLite db, dashboard/ exports)
├── notebooks/
│   ├── 01_eda.ipynb               # Target dist, drivers, MRR at risk
│   ├── 02_baseline_model.ipynb    # Logistic regression
│   ├── 03_xgboost.ipynb           # XGBoost + LR comparison
│   ├── 04_cost_analysis.ipynb     # Optimal threshold + sensitivity
│   ├── 05_dashboard_export.ipynb  # CV predict → SQLite → CSV exports
│   └── 06_retention_strategies.ipynb  # Strategy ROI ranking
├── sql/
│   ├── 01_feature_engineering.sql # customer_features view
│   └── 02_at_risk_cohorts.sql     # cohort, summary, churn-rate matrix views
├── src/
│   ├── config.py                  # Paths, column lists, random seed
│   ├── data_loader.py             # kagglehub download + cleaning
│   ├── features.py                # One-hot encoding + train/test split
│   ├── model.py                   # LR + XGBoost trainers (imbalance-aware)
│   ├── evaluation.py              # Metrics, ROC, confusion, importances
│   └── cost_analysis.py           # Cost-sensitive policy framework
├── scripts/
│   └── compute_headline_metrics.py  # Reruns the pipeline and prints README values
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/BK5102/Churn-Predictor_-Revenue-Impact-Engine.git
cd Churn-Predictor_-Revenue-Impact-Engine

# Set up virtual environment
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install kagglehub               # not pinned in requirements.txt (see data/README.md)

# Configure Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars)

# Launch Jupyter and run the notebooks in order
jupyter notebook
```

### Run order

```
01_eda → 02_baseline_model → 03_xgboost → 04_cost_analysis → 05_dashboard_export → 06_retention_strategies
```

Each notebook is self-contained — you can re-run any one of them without re-executing the earlier ones, since the `src/` modules handle loading and featurization deterministically.

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SQLite · SQL · kagglehub · Jupyter

---

## Author

**Bhavana Kannan**
