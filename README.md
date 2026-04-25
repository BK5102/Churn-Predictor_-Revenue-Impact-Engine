# Churn Predictor + Revenue Impact Engine

> Predicting customer churn for a telecom provider and quantifying the revenue saved by targeted retention interventions.

**Status:** ✅ Pipeline complete. Headline numbers below are populated by running the notebooks end-to-end against the Kaggle dataset.

---

## The Business Problem

Customer churn is expensive. For a subscription business, losing a customer means losing not just this month's revenue but the entire remaining customer lifetime. But retention interventions also cost money — discounts, support calls, and contract incentives eat into margin. The question isn't *"can we predict churn?"* — it's *"at what churn probability does intervention pay for itself, and which intervention pays for itself fastest?"*

This project answers those questions using the Telco Customer Churn dataset, a standard benchmark of ~7,000 telecom customers with subscription details and churn labels.

## Key Questions

1. Which customers are most likely to churn, and what drives their risk?
2. How much monthly recurring revenue (MRR) is at risk?
3. At what predicted churn probability should the business intervene, given the cost and success rate of interventions?
4. Among possible retention strategies (discounts, contract upgrades, premium support), which has the best ROI?

## Headline Results

> Numbers below are produced by `scripts/compute_headline_metrics.py` running the full pipeline against the Kaggle dataset. Reproduce with `python scripts/compute_headline_metrics.py`.

| Metric | Value | Source |
|---|---|---|
| Customer base | 7,032 (after dropping 11 tenure=0 rows) | `data_loader.load_clean()` |
| Overall churn rate | **26.6%** | `01_eda.ipynb` |
| Logistic regression test AUC | **0.835** | `02_baseline_model.ipynb` |
| XGBoost test AUC | **0.830** | `03_xgboost.ipynb` |
| Total MRR (snapshot) | **$455,661** | `01_eda.ipynb` § 5 |
| 12-month MRR from actual churners | **$1,669,570** | `01_eda.ipynb` § 5 |
| Optimal intervention threshold τ* | **0.10** (default assumptions) | `04_cost_analysis.ipynb` |
| Lift vs. do-nothing baseline (test set) | **+$144,379** over 12 months | `04_cost_analysis.ipynb` |
| Top strategy by **net value** | Combined discount + addon (ROI 1.42×) | `06_retention_strategies.ipynb` |
| Top strategy by **ROI** | Free security addon, 6mo (ROI 2.56×) | `06_retention_strategies.ipynb` |

**Note on AUC:** logistic regression edges out XGBoost by 0.005 here — meaningfully tied at this dataset size. With ~7k rows and well-separated bucketed features, the linear model captures most of the signal. XGBoost would likely pull ahead on a larger dataset or with richer interaction terms; for this scale, the more interpretable LR is a defensible production choice.

**Note on the τ\* = 0.10:** the optimal threshold falls below the modeling default of 0.5 because the average CLV ($65/mo × 12 = $780) is large relative to the $50 intervention cost — even modestly-likely churners are worth targeting. The notebook's sensitivity heatmap shows τ\* climbing above 0.5 only when intervention cost exceeds $150 or success rate falls below 15%.

**Note on the negative `net_value` numbers in the strategy table:** `net_value = saved_revenue − intervention_costs − unrecovered_churn_losses`, so it is realized total profit (negative because we can't save 100% of churners), not delta vs. baseline. The headline number to quote is **lift vs. do-nothing** (+$144K on the test set, ≈ +$720K extrapolated to the full base over 12 months).

## Dashboard

Tableau-ready CSVs are produced by `notebooks/05_dashboard_export.ipynb` into `data/dashboard/`:

- `at_risk_cohorts.csv` — per-customer scores tagged with risk × revenue priority bucket
- `cohort_summary.csv` — KPI aggregates per priority bucket
- `churn_by_tenure_contract.csv` — churn-rate matrix for the headline heatmap
- `customer_features.csv` — engineered features (tenure bucket, MRR tier, add-on counts)

### Building the workbook

1. Run `notebooks/05_dashboard_export.ipynb` — generates the four CSVs above.
2. In Tableau Public Desktop, connect to `at_risk_cohorts.csv` as the primary data source; add the others as secondary sources joined on `priority_bucket` / `tenure_bucket`.
3. Recommended dashboard layout:
   - **Top row (KPI cards):** customer base, total MRR, expected revenue at risk, customers targeted at τ\*.
   - **Cohort treemap:** `priority_bucket` × `MonthlyCharges` (size) × `churn_probability` (color).
   - **Heatmap:** churn rate by `tenure_bucket × Contract` (the EDA's clearest segmentation).
   - **Action table:** top-N highest `expected_revenue_at_risk` customers with `Contract` + `PaymentMethod` for action recommendations.
4. Publish to Tableau Public and replace the line below.

*Tableau Public link: TBD — link will be added here once the workbook is published.*

---

## Methodology

### Data
- **Source:** [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers, 21 features
- **Target:** Binary `Churn` label
- **Loading:** via `kagglehub` (see `data/README.md`); the loader handles the known `TotalCharges` blank-row quirk for tenure=0 customers.

### Pipeline
1. **EDA** (`01_eda.ipynb`) — churn rate, feature-target relationships, MRR at risk.
2. **Baseline model** (`02_baseline_model.ipynb`) — logistic regression with `class_weight="balanced"` to handle the 73/27 split.
3. **Tree model** (`03_xgboost.ipynb`) — XGBoost with `scale_pos_weight`, head-to-head vs LR on the same split.
4. **Cost-sensitive analysis** (`04_cost_analysis.ipynb`) — sweep thresholds, find the τ that maximizes net realized value, run a 5×5 sensitivity heatmap over `intervention_cost × success_rate`.
5. **SQL feature engineering** (`sql/01_feature_engineering.sql`, `sql/02_at_risk_cohorts.sql`) — analyst-facing views: bucketed features, priority cohorts, KPI aggregates.
6. **Dashboard export** (`05_dashboard_export.ipynb`) — orchestrates everything: out-of-fold CV predictions for every customer, SQLite load, runs the SQL files, exports Tableau-ready CSVs.
7. **Strategy ranking** (`06_retention_strategies.ipynb`) — each retention play (discount / addon / contract upgrade / premium support / combined) becomes a (cost, success_rate) tuple. Each strategy gets its own optimal threshold; we compare on both net value and ROI, with a ±30% sensitivity check on the success-rate assumption.

### Key Assumptions (Cost-Sensitive Layer)

These are placeholders for the user to override with real ops data. They live in `src/cost_analysis.CostAssumptions`:

| Parameter | Default | Rationale |
|---|---|---|
| Intervention cost | $50 / targeted customer | Approximate cost of a retention call plus a small discount |
| Intervention success rate | 30% | Industry-typical for outbound retention; should be measured via A/B test |
| CLV horizon | 12 months of MRR | Simple horizon-based CLV; decoupled from threshold sensitivity |

The decision rule reduces to: **intervene iff `P(churn) × success_rate × CLV > intervention_cost`**.

### Class imbalance and metric choice

Churn is ~27% positive. Accuracy is misleading at this rate (a model that always predicts "stay" still scores 73%). Reported metrics lead with **AUC** and **recall on the churn class**; the cost-sensitive analysis takes precedence over any metric for the deployment decision.

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

## Roadmap

- [x] **Week 1 — Foundation:** EDA, baseline model, XGBoost comparison
- [x] **Week 2 — Business Layer:** Cost-sensitive analysis, optimal threshold
- [x] **Week 3 — Pipeline + Dashboard:** SQL feature engineering, Tableau-ready exports
- [x] **Week 4 — Polish:** Retention strategy ranking, README, feature importance

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SQLite · SQL · kagglehub · Tableau Public · Jupyter

---

## Author

**Bhavana Kannan** · [LinkedIn](#) · [Portfolio](#)

*Replace the `#` placeholders above with your LinkedIn and portfolio URLs.*
