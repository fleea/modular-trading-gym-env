import pandas as pd


def filter_noise(df: pd.DataFrame, threshold: float, col_name: str) -> pd.DataFrame:
    """
    Filters out noise from a DataFrame based on a percentage change threshold.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The percentage change threshold to filter noise.
    col_name (str): The column name on which to calculate the percentage change.

    Returns:
    pd.DataFrame: The filtered DataFrame with noise removed.
    """
    # Create a copy of the dataframe to avoid modifying the original
    filtered_df = df.copy()

    # Calculate the percentage change of the specified column
    pct_change = filtered_df[col_name].pct_change().abs() * 100

    # Create a mask for values that exceed the threshold or are the first row
    mask = (pct_change > threshold) | (pct_change.isna())

    # Apply the mask to filter rows
    filtered_df = filtered_df[mask]

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df
