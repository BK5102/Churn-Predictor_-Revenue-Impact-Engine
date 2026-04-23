"""Load the Telco Customer Churn dataset via kagglehub and do minimal cleaning.

The raw CSV has one quirk worth handling at load time: `TotalCharges` is a
string column with blank values (" ") for customers whose `tenure` is 0. We
coerce it to numeric and drop those rows — they represent brand-new customers
with no billing history and can't churn by definition in this snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CSV_FILENAME, KAGGLE_DATASET


def download_dataset() -> Path:
    """Download the Telco dataset via kagglehub and return the CSV path.

    Requires ``kagglehub`` (``pip install kagglehub``) and Kaggle credentials
    in ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``
    environment variables. See ``data/README.md``.
    """
    import kagglehub  # Imported lazily so the rest of src/ doesn't require it.

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_path = dataset_dir / CSV_FILENAME
    if not csv_path.exists():
        # Fall back to whatever CSV is in the cached directory
        csvs = list(dataset_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {dataset_dir}")
        csv_path = csvs[0]
    return csv_path


def load_raw(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV with no cleaning applied."""
    if csv_path is None:
        csv_path = download_dataset()
    return pd.read_csv(csv_path)


def load_clean(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV and apply minimal, dataset-specific cleaning.

    - Coerce ``TotalCharges`` from str to float (blanks become NaN).
    - Drop rows where ``TotalCharges`` is NaN (tenure=0 customers).
    - Cast ``SeniorCitizen`` from 0/1 int to ``Yes``/``No`` for consistency
      with the other binary categoricals.
    """
    df = load_raw(csv_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df
