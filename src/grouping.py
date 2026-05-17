from src.config import EFFICIENCY_COLUMN, FEATURE_COLUMNS


def group_conditions(data):
    grouped = (
        data.groupby(FEATURE_COLUMNS, as_index=False)
        .agg(
            count=(EFFICIENCY_COLUMN, "count"),
            mean_efficiency=(EFFICIENCY_COLUMN, "mean"),
            max_efficiency=(EFFICIENCY_COLUMN, "max"),
            std_efficiency=(EFFICIENCY_COLUMN, "std"),
            min_efficiency=(EFFICIENCY_COLUMN, "min"),
            median_efficiency=(EFFICIENCY_COLUMN, "median"),
        )
        .sort_values("max_efficiency", ascending=False)
        .reset_index(drop=True)
    )

    return grouped
