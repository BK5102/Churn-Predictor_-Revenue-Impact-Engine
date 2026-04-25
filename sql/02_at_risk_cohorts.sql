-- =============================================================================
-- 02_at_risk_cohorts.sql
--
-- Joins customer features with model predictions to produce dashboard-ready
-- cohort views. Every customer is tagged with a priority bucket combining
-- predicted churn probability and revenue (MRR), so the retention team can
-- target the highest-value at-risk customers first.
--
-- Prerequisites:
--   - `customer_features` view (sql/01_feature_engineering.sql)
--   - `predictions` table with: customerID, churn_probability, is_targeted
--     populated by notebooks/05_dashboard_export.ipynb
-- =============================================================================

DROP VIEW IF EXISTS at_risk_cohorts;

CREATE VIEW at_risk_cohorts AS
SELECT
    cf.customerID,
    cf.tenure_bucket,
    cf.mrr_tier,
    cf.Contract,
    cf.PaymentMethod,
    cf.InternetService,
    cf.MonthlyCharges,
    cf.clv_12mo,
    cf.has_protection_addon,
    cf.total_addons,
    cf.is_month_to_month,
    cf.pays_by_echeck,
    p.churn_probability,
    p.is_targeted,
    cf.Churn AS actual_churn,

    -- Risk × revenue priority — analyst-readable bucket. The thresholds
    -- (0.5 / 0.3 for risk, $70 for MRR) come from the optimal-threshold
    -- analysis (notebook 04) and the EDA's MRR distribution.
    CASE
        WHEN p.churn_probability >= 0.5 AND cf.MonthlyCharges >= 70 THEN '1-priority-high-risk-high-mrr'
        WHEN p.churn_probability >= 0.5 AND cf.MonthlyCharges <  70 THEN '2-priority-high-risk-low-mrr'
        WHEN p.churn_probability >= 0.3 AND cf.MonthlyCharges >= 70 THEN '3-priority-mid-risk-high-mrr'
        WHEN p.churn_probability >= 0.3                              THEN '4-priority-mid-risk-low-mrr'
        ELSE '5-monitor'
    END AS priority_bucket,

    -- Expected revenue at risk = P(churn) * 12-month CLV
    p.churn_probability * cf.clv_12mo AS expected_revenue_at_risk

FROM customer_features cf
JOIN predictions p ON cf.customerID = p.customerID;


-- -----------------------------------------------------------------------------
-- Headline aggregations for the dashboard's top-row KPI cards
-- -----------------------------------------------------------------------------

DROP VIEW IF EXISTS cohort_summary;

CREATE VIEW cohort_summary AS
SELECT
    priority_bucket,
    COUNT(*)                          AS n_customers,
    SUM(MonthlyCharges)               AS total_mrr,
    SUM(expected_revenue_at_risk)     AS expected_revenue_at_risk,
    AVG(churn_probability)            AS avg_churn_probability,
    SUM(is_targeted)                  AS n_targeted,
    SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) AS n_actual_churners
FROM at_risk_cohorts
GROUP BY priority_bucket
ORDER BY priority_bucket;


-- -----------------------------------------------------------------------------
-- Tableau-friendly slice: churn-rate matrix by tenure bucket × contract
-- -----------------------------------------------------------------------------

DROP VIEW IF EXISTS churn_by_tenure_contract;

CREATE VIEW churn_by_tenure_contract AS
SELECT
    tenure_bucket,
    Contract,
    COUNT(*) AS n_customers,
    AVG(CASE WHEN actual_churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS actual_churn_rate,
    AVG(churn_probability) AS predicted_churn_rate,
    SUM(MonthlyCharges) AS total_mrr
FROM at_risk_cohorts
GROUP BY tenure_bucket, Contract
ORDER BY tenure_bucket, Contract;
