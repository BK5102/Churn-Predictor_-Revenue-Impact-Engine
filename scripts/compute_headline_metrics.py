"""One-shot script to populate the README's Headline Results table.

Runs the same logic the notebooks do, prints the numbers, exits. Not
imported anywhere — kept for repeatability so the README values can be
refreshed when assumptions change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.cost_analysis import CostAssumptions, optimal_threshold, sweep_thresholds
from src.data_loader import load_clean
from src.features import make_features, train_test_split_scaled
from src.model import train_logistic_regression, train_xgboost


@dataclass(frozen=True)
class Strategy:
    name: str
    cost: float
    success_rate: float


def main() -> None:
    print("Loading data...")
    df = load_clean()
    X, y = make_features(df)
    print(f"  {len(df):,} customers, churn rate = {y.mean():.4f}")

    # 1. Churn rate + MRR at risk (raw) -------------------------------------
    churn_rate = y.mean()
    total_mrr = df["MonthlyCharges"].sum()
    mrr_from_actual_churners = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
    mrr_at_risk_12mo = mrr_from_actual_churners * 12

    # 2. LR vs XGB AUC on the held-out test split --------------------------
    X_train, X_test, y_train, y_test = train_test_split_scaled(X, y, scale=True)
    monthly_charges_test = df.loc[X_test.index, "MonthlyCharges"].values

    lr = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    # 3. Optimal threshold via cost-sensitive sweep on the test set --------
    from src.cost_analysis import policy_value
    assumptions = CostAssumptions(intervention_cost=50.0, success_rate=0.30, clv_horizon_months=12)
    sweep = sweep_thresholds(y_test, xgb_proba, monthly_charges_test, assumptions)
    best = optimal_threshold(sweep)
    # Baselines for context: do-nothing target nobody, treat-everyone target all
    do_nothing = policy_value(y_test, xgb_proba, monthly_charges_test, 1.01, assumptions)
    treat_all  = policy_value(y_test, xgb_proba, monthly_charges_test, 0.0,  assumptions)
    lift_vs_donothing = best["net_value"] - do_nothing["net_value"]

    # 4. Strategy ranking on the same test set -----------------------------
    strategies = [
        Strategy("premium_support_outreach",     25.0, 0.20),
        Strategy("free_security_addon_6mo",      20.0, 0.22),
        Strategy("discount_10pct_6mo",           40.0, 0.28),
        Strategy("contract_upgrade_incentive",   50.0, 0.40),
        Strategy("discount_plus_addon_combined", 60.0, 0.45),
    ]
    rows = []
    for s in strategies:
        a = CostAssumptions(intervention_cost=s.cost, success_rate=s.success_rate, clv_horizon_months=12)
        sw = sweep_thresholds(y_test, xgb_proba, monthly_charges_test, a)
        b = optimal_threshold(sw)
        saved = b["saved_revenue"]
        spent = b["intervention_costs"]
        rows.append({
            "name": s.name,
            "tau": b["threshold"],
            "net_value": b["net_value"],
            "roi": (saved - spent) / spent if spent > 0 else float("nan"),
        })
    by_value = max(rows, key=lambda r: r["net_value"])
    by_roi = max(rows, key=lambda r: r["roi"])

    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    print(f"Overall churn rate:               {churn_rate:.1%}")
    print(f"LR  test AUC:                     {lr_auc:.3f}")
    print(f"XGB test AUC:                     {xgb_auc:.3f}")
    print(f"Total MRR (snapshot):             ${total_mrr:>12,.0f}")
    print(f"12-mo MRR from actual churners:   ${mrr_at_risk_12mo:>12,.0f}")
    print(f"Optimal threshold tau* (default assumptions): {best['threshold']:.2f}")
    print(f"  Net value at tau*:              ${best['net_value']:>12,.0f}")
    print(f"  Net value (do-nothing):         ${do_nothing['net_value']:>12,.0f}")
    print(f"  Net value (treat-everyone):     ${treat_all['net_value']:>12,.0f}")
    print(f"  Lift vs do-nothing (test set):  ${lift_vs_donothing:>12,.0f}")
    print(f"  Customers targeted at tau*:     {int(best['n_targeted']):>12,}")
    print()
    print("Strategy ranking (test set):")
    for r in sorted(rows, key=lambda r: r["net_value"], reverse=True):
        print(f"  {r['name']:<35} tau*={r['tau']:.2f}  net=${r['net_value']:>9,.0f}  ROI={r['roi']:.2f}x")
    print()
    print(f"Top by net value: {by_value['name']}  (${by_value['net_value']:,.0f}, ROI {by_value['roi']:.2f}x)")
    print(f"Top by ROI:       {by_roi['name']}  (${by_roi['net_value']:,.0f}, ROI {by_roi['roi']:.2f}x)")


if __name__ == "__main__":
    main()
