import pandas as pd


def augment_col_difference(
    df: pd.DataFrame, col_names: list, window: int = 20
) -> pd.DataFrame:
    """
    Augment the dataframe with the difference between consecutive rows for multiple columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    col_names (list): List of column names to compute differences for

    Returns:
    pd.DataFrame: DataFrame with new columns added for differences
    """
    # Create a copy of the input DataFrame
    df_result = df.copy()

    # Compute differences for all specified columns at once
    differences = df[col_names].diff()

    # Add new columns to the DataFrame
    for col in col_names:
        col_change = f"{col}_change"
        df_result[col_change] = differences[col]
        # Compute rolling standard deviation on the change column
        df_result[f"{col_change}_std"] = (
            df_result[col_change].rolling(window=window, min_periods=1).std()
        )

    # Add the new columns to the DataFrame
    return df_result
