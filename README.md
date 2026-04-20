# Churn Predictor + Revenue Impact Engine

> Predicting customer churn for a telecom provider and quantifying the revenue saved by targeted retention interventions.

**Status:** 🚧 In progress — see [Roadmap](#roadmap) below.

---

## The Business Problem

Customer churn is expensive. For a subscription business, losing a customer means losing not just this month's revenue but the entire remaining customer lifetime. But retention interventions also cost money — discounts, support calls, and contract incentives eat into margin. The question isn't *"can we predict churn?"* — it's *"at what churn probability does intervention pay for itself?"*

This project answers that question using the Telco Customer Churn dataset, a standard benchmark of ~7,000 telecom customers with subscription details and churn labels.

## Key Questions

1. Which customers are most likely to churn, and what drives their risk?
2. How much monthly recurring revenue (MRR) is at risk?
3. At what predicted churn probability should the business intervene, given the cost and success rate of interventions?
4. Among possible retention strategies (discounts, contract upgrades, premium support), which has the best ROI?

## Headline Results

*To be filled in as the project progresses.*

- Model AUC: TBD
- MRR at risk: TBD
- Optimal intervention threshold: TBD
- Top retention strategy by ROI: TBD

## Dashboard

*Tableau Public link coming in Week 3.*

---

## Methodology

### Data
- **Source:** [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers, 21 features
- **Target:** Binary `Churn` label

### Pipeline
1. **EDA** — churn rate, feature-target relationships, revenue distribution
2. **Feature engineering** — encoded in SQL (`sql/`) to mirror real analyst workflows
3. **Modeling** — logistic regression baseline + XGBoost
4. **Cost-sensitive analysis** — expected value across probability thresholds
5. **Strategy ranking** — ROI estimates for each intervention type
6. **Dashboard** — at-risk cohorts, revenue impact, recommended actions

### Key Assumptions (Cost-Sensitive Layer)
*To be documented in Week 2 after EDA informs realistic values.*

- Customer Lifetime Value (CLV): TBD
- Retention intervention cost: TBD
- Intervention success rate: TBD

---

## Project Structure

```
churn-predictor/
├── data/              # Raw data (gitignored — see data/README.md)
├── notebooks/         # Jupyter notebooks, numbered by workflow order
├── sql/               # Feature engineering and data prep queries
├── src/               # Reusable Python modules
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd churn-predictor

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle into data/
# (see data/README.md for instructions)

# Launch Jupyter
jupyter notebook
```

---

## Roadmap

- [ ] **Week 1 — Foundation:** EDA, baseline model, XGBoost comparison
- [ ] **Week 2 — Business Layer:** Cost-sensitive analysis, optimal threshold
- [ ] **Week 3 — Pipeline + Dashboard:** SQL feature engineering, Tableau dashboard
- [ ] **Week 4 — Polish:** Retention strategy ranking, README, feature importance

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SQLite · SQL · Tableau Public · Jupyter

---

## Author

*Your name · [LinkedIn](#) · [Portfolio](#)*
