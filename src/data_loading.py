import pandas as pd

from src.config import (
    EFFICIENCY_COLUMN,
    FEATURE_COLUMNS,
    ORIGINAL_SPIRO_COLUMN,
    RAW_DATA_PATH,
    RENAMED_SPIRO_COLUMN,
)


def load_raw_dataset(path=RAW_DATA_PATH):
    """Load the raw dataset and normalize column names used by the workflow."""
    data = pd.read_csv(path)
    renamed_spiro_column = False

    if ORIGINAL_SPIRO_COLUMN in data.columns:
        data = data.rename(columns={ORIGINAL_SPIRO_COLUMN: RENAMED_SPIRO_COLUMN})
        renamed_spiro_column = True

    required_columns = [EFFICIENCY_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s) after loading data: {missing}")

    return data, renamed_spiro_column


def filter_by_efficiency_threshold(data, threshold):
    return data[data[EFFICIENCY_COLUMN] >= threshold].copy()
