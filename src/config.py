"""Project-wide configuration: paths, constants, model defaults."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = PROJECT_ROOT / "sql"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "blastchar/telco-customer-churn"
CSV_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"

# Columns that are categorical even though some arrive as int/str
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_CATEGORICAL = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
