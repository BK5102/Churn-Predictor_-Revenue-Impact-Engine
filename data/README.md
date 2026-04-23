# Data

Raw data lives in this directory and is **gitignored** (`data/*.csv`). Only this README is tracked.

## Dataset: Telco Customer Churn

- **Source:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 rows, 21 columns
- **Target:** `Churn` (Yes / No)
- **Expected filename:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Download via kagglehub

`kagglehub` is not yet pinned in `requirements.txt`. Install it first:

```bash
pip install kagglehub
```

Then download the dataset:

```python
import kagglehub

# Downloads the latest version to the local kagglehub cache
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Path to dataset files:", path)
```

`path` points to the cached directory (e.g. `~/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/N/`). The loader in `src/data_loader.py` resolves this path and reads the CSV directly — you do **not** need to copy the file into `data/`.

## Authentication

`kagglehub` uses Kaggle credentials from either:
- `~/.kaggle/kaggle.json` (download from Kaggle → Account → Create New API Token), or
- Environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Schema

| Column | Type | Notes |
|---|---|---|
| `customerID` | str | Drop before modeling |
| `gender`, `Partner`, `Dependents` | categorical (Yes/No) | |
| `SeniorCitizen` | int (0/1) | Semantically categorical |
| `tenure` | int | Months as a customer |
| `PhoneService`, `MultipleLines` | categorical | |
| `InternetService` | categorical (DSL / Fiber / No) | |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical | `No internet service` is a distinct level |
| `Contract` | categorical (Month-to-month / One year / Two year) | |
| `PaperlessBilling` | categorical | |
| `PaymentMethod` | categorical | |
| `MonthlyCharges` | float | USD |
| `TotalCharges` | float* | Stored as str with blanks for tenure=0 rows; coerce to float and drop/impute |
| `Churn` | categorical (Yes/No) | Target |
