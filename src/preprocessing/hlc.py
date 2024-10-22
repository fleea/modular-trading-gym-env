import pandas as pd
from src.utils.fraction import calculate_fraction
from src.utils.rms import get_rms_multiplier


# Add columns in data pre-processing
change = "change_"
high = "high"
low = "low"
close = "close"
d = "prev_day"
w = "prev_week"
m = "prev_month"
d_high = d + "_" + high
d_low = d + "_" + low
d_close = d + "_" + close
w_high = w + "_" + high
w_low = w + "_" + low
w_close = w + "_" + close
m_high = m + "_" + high
m_low = m + "_" + low
m_close = m + "_" + close


def augment_with_hlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment OHLCV data with previous day's, week's, and month's high, low, and close values.

    :param df: A pandas DataFrame containing OHLCV data with a DatetimeIndex.
    :return: DataFrame augmented with previous day, week, and month's HLC values.
    """

    # Ensure the dataframe has a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be of type datetime64.")

    # Resample the data to daily, weekly, and monthly frequencies

    daily = (
        df.resample("D")
        .agg({"high": "max", "low": "min", "close": "last"})
        .shift(1)
        .ffill()
    )
    weekly = (
        df.resample("W").agg({"high": "max", "low": "min", "close": "last"}).ffill()
    )
    monthly = (
        df.resample("MS")
        .agg({"high": "max", "low": "min", "close": "last"})
        .shift(1)
        .ffill()
    )

    # Rename columns to avoid collisions

    daily.columns = [d_high, d_low, d_close]
    weekly.columns = [w_high, w_low, w_close]
    monthly.columns = [m_high, m_low, m_close]

    # Forward fill the daily, weekly, and monthly data to align with the hourly data
    daily_filled = daily.reindex(df.index, method="ffill")
    weekly_filled = weekly.reindex(df.index, method="ffill")
    monthly_filled = monthly.reindex(df.index, method="ffill")

    # Add day of the week and month as index
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Concatenate the original dataframe with the augmented data
    df_augmented = pd.concat([df, daily_filled, weekly_filled, monthly_filled], axis=1)

    # # Create a vectorized version of calculate_fraction
    # vectorized_calculate_fraction = np.vectorize(calculate_fraction)

    # Define time periods and corresponding columns
    time_periods = [d, w, m]
    metrics = [high, low, close]

    for period in time_periods:
        for metric in metrics:
            prev_metric = period + "_" + metric
            change_fraction = calculate_fraction(df_augmented[metric], df_augmented[prev_metric])
            rms_multiplier = get_rms_multiplier(change_fraction)
            # print(f"rms_multiplier_{period}_{metric}: {rms_multiplier}")
            percentage_change = change_fraction * rms_multiplier
            df_augmented[f'{change}{prev_metric}'] = percentage_change

    return df_augmented
