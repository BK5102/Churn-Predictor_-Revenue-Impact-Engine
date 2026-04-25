"""Cost-sensitive analysis: turn churn probabilities into business decisions.

The model gives us ``P(churn)`` per customer. The business question is *not*
"who churns?" but "for whom does intervention pay for itself?". Answering
that requires three numbers we don't get from the model:

- ``intervention_cost`` — per-customer cost of running the retention play
  (discount, support call, contract incentive). Default $50.
- ``success_rate`` — probability the intervention prevents churn given the
  customer would have churned otherwise. Default 0.30 (industry-typical for
  retention calls; revisit with A/B test data when available).
- ``clv_horizon_months`` — how many months of MRR we credit to a saved
  customer. Default 12. A longer horizon makes intervention look better.

The decision rule is simple: intervene iff
``p * success_rate * CLV > intervention_cost``.
This module backtests that rule across thresholds on a labeled test set so
we can see which threshold maximizes realized $.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostAssumptions:
    """Bundle the business parameters so they're passed around explicitly."""

    intervention_cost: float = 50.0
    success_rate: float = 0.30
    clv_horizon_months: int = 12


def customer_clv(
    monthly_charges: pd.Series | np.ndarray,
    horizon_months: int = 12,
) -> np.ndarray:
    """Estimate per-customer CLV as ``MonthlyCharges * horizon_months``.

    This is intentionally simple. A more sophisticated CLV would account for
    expected remaining tenure, gross margin, and discount rate. We use the
    flat horizon so threshold sensitivity (Week 2) doesn't get tangled up
    with CLV-modeling assumptions.
    """
    return np.asarray(monthly_charges) * horizon_months


def policy_value(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    monthly_charges: pd.Series | np.ndarray,
    threshold: float,
    assumptions: CostAssumptions,
) -> dict[str, float]:
    """Total realized value of the policy "intervene iff p >= threshold".

    Accounting (per customer):
    - Targeted (p >= threshold):
        * Pay ``intervention_cost`` always.
        * If they would have churned (``y_true == 1``): with probability
          ``success_rate`` we save the customer and recover their CLV.
    - Not targeted (p < threshold):
        * Pay nothing.
        * If they would have churned: lose their CLV (counted as negative).

    Returns a dict so it's easy to slot into a DataFrame row.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    clv = customer_clv(monthly_charges, assumptions.clv_horizon_months)

    targeted = y_proba >= threshold
    not_targeted = ~targeted

    # Costs / benefits
    intervention_costs = targeted.sum() * assumptions.intervention_cost

    saved_revenue = (
        (targeted & (y_true == 1)) * clv * assumptions.success_rate
    ).sum()

    lost_revenue = ((not_targeted & (y_true == 1)) * clv).sum()
    # Also: customers we targeted who would have churned but the intervention
    # *failed* — we still lose (1 - success_rate) * CLV for them.
    lost_revenue += (
        (targeted & (y_true == 1)) * clv * (1 - assumptions.success_rate)
    ).sum()

    net_value = saved_revenue - intervention_costs - lost_revenue

    return {
        "threshold": threshold,
        "n_targeted": int(targeted.sum()),
        "intervention_costs": float(intervention_costs),
        "saved_revenue": float(saved_revenue),
        "lost_revenue": float(lost_revenue),
        "net_value": float(net_value),
    }


def sweep_thresholds(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    monthly_charges: pd.Series | np.ndarray,
    assumptions: CostAssumptions,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate ``policy_value`` across many thresholds; return a tidy df."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    rows = [
        policy_value(y_true, y_proba, monthly_charges, t, assumptions)
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def optimal_threshold(sweep_df: pd.DataFrame) -> dict[str, float]:
    """Return the row of ``sweep_df`` with the highest ``net_value``."""
    best = sweep_df.loc[sweep_df["net_value"].idxmax()]
    return best.to_dict()
