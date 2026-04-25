-- =============================================================================
-- 01_feature_engineering.sql
--
-- Derives modeling- and dashboard-friendly columns from the raw `customers`
-- table. This view is the analyst-facing equivalent of the one-hot encoding
-- and bucketing the Python pipeline does in src/features.py — keeping a
-- SQL version means anyone with database access can answer "how many
-- month-to-month fiber customers do we have?" without touching Python.
--
-- Prerequisites: a `customers` table populated from the Telco CSV.
-- Compatible with: SQLite (uses standard CASE WHEN, no proprietary syntax).
-- =============================================================================

DROP VIEW IF EXISTS customer_features;

CREATE VIEW customer_features AS
SELECT
    customerID,

    -- Pass-through columns the dashboard / downstream views still need
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    Contract,
    PaymentMethod,
    InternetService,
    PaperlessBilling,
    MonthlyCharges,
    TotalCharges,
    Churn,

    -- Tenure bucket — the EDA showed first-year customers churn far more
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_bucket,

    -- MRR tier — supports revenue-weighted cohort views in the dashboard
    CASE
        WHEN MonthlyCharges <  35 THEN 'low'
        WHEN MonthlyCharges <  70 THEN 'mid'
        WHEN MonthlyCharges < 100 THEN 'high'
        ELSE 'premium'
    END AS mrr_tier,

    -- Count of optional add-on services the customer subscribes to.
    -- "No internet service" rows correctly resolve to 0 here because
    -- they're not equal to 'Yes'.
    (
        CASE WHEN OnlineSecurity   = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN OnlineBackup     = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN TechSupport      = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingTV      = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingMovies  = 'Yes' THEN 1 ELSE 0 END
    ) AS total_addons,

    -- Any protection / support add-on (these correlate with retention in EDA)
    CASE
        WHEN OnlineSecurity   = 'Yes'
          OR OnlineBackup     = 'Yes'
          OR DeviceProtection = 'Yes'
          OR TechSupport      = 'Yes'
        THEN 1 ELSE 0
    END AS has_protection_addon,

    -- Simple 12-month CLV estimate (matches src/cost_analysis.customer_clv)
    MonthlyCharges * 12 AS clv_12mo,

    -- Top-3 churn-driver flags from the model (week 1 finding)
    CASE WHEN Contract = 'Month-to-month'      THEN 1 ELSE 0 END AS is_month_to_month,
    CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END AS pays_by_echeck,
    CASE WHEN InternetService = 'Fiber optic'    THEN 1 ELSE 0 END AS has_fiber

FROM customers;
