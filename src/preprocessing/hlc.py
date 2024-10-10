import pandas as pd
import numpy as np
from src.utils.fraction import calculate_fraction
from src.utils.rms import get_rms_multiplier


# Add columns in data pre-processing
change = "change_"
d_high = "prev_day_high"
d_low = "prev_day_low"
d_close = "prev_day_close"
w_high = "prev_week_high"
w_low = "prev_week_low"
w_close = "prev_week_close"
m_high = "prev_month_high"
m_low = "prev_month_low"
m_close = "prev_month_close"


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

    # Create a vectorized version of calculate_fraction
    vectorized_calculate_fraction = np.vectorize(calculate_fraction)

    # DAY

    change_d_high = vectorized_calculate_fraction(df_augmented['high'], df_augmented[d_high])
    rms_multiplier_d_high = get_rms_multiplier(change_d_high)
    df_augmented[f'{change}{d_high}'] = change_d_high * rms_multiplier_d_high

    change_d_low = vectorized_calculate_fraction(df_augmented['low'], df_augmented[d_low])
    rms_multiplier_d_low = get_rms_multiplier(change_d_low)
    print(f"rms_multiplier_d_low: {rms_multiplier_d_low}")
    df_augmented[f'{change}{d_low}'] = change_d_low * rms_multiplier_d_low

    change_d_close = vectorized_calculate_fraction(df_augmented['close'], df_augmented[d_close])
    rms_multiplier_d_close = get_rms_multiplier(change_d_close)
    print(f"rms_multiplier_d_close: {rms_multiplier_d_close}")
    df_augmented[f'{change}{d_close}'] = change_d_close * rms_multiplier_d_close

    # WEEK
    change_w_high = vectorized_calculate_fraction(df_augmented['high'], df_augmented[w_high])
    rms_multiplier_w_high = get_rms_multiplier(change_w_high)
    print(f"rms_multiplier_w_high: {rms_multiplier_w_high}")
    df_augmented[f'{change}{w_high}'] = change_w_high * rms_multiplier_w_high

    change_w_low = vectorized_calculate_fraction(df_augmented['low'], df_augmented[w_low])
    rms_multiplier_w_low = get_rms_multiplier(change_w_low)
    print(f"rms_multiplier_w_low: {rms_multiplier_w_low}")
    df_augmented[f'{change}{w_low}'] = change_w_low * rms_multiplier_w_low

    change_w_close = vectorized_calculate_fraction(df_augmented['close'], df_augmented[w_close])
    rms_multiplier_w_close = get_rms_multiplier(change_w_close)
    print(f"rms_multiplier_w_close: {rms_multiplier_w_close}")
    df_augmented[f'{change}{w_close}'] = change_w_close * rms_multiplier_w_close

    # MONTH
    change_m_high = vectorized_calculate_fraction(df_augmented['high'], df_augmented[m_high])
    rms_multiplier_m_high = get_rms_multiplier(change_m_high)
    print(f"rms_multiplier_m_high: {rms_multiplier_m_high}")
    df_augmented[f'{change}{m_high}'] = change_m_high * rms_multiplier_m_high

    change_m_low = vectorized_calculate_fraction(df_augmented['low'], df_augmented[m_low])
    rms_multiplier_m_low = get_rms_multiplier(change_m_low)
    print(f"rms_multiplier_m_low: {rms_multiplier_m_low}")
    df_augmented[f'{change}{m_low}'] = change_m_low * rms_multiplier_m_low

    change_m_close = vectorized_calculate_fraction(df_augmented['close'], df_augmented[m_close])    
    rms_multiplier_m_close = get_rms_multiplier(change_m_close)
    print(f"rms_multiplier_m_close: {rms_multiplier_m_close}")
    df_augmented[f'{change}{m_close}'] = change_m_close * rms_multiplier_m_close

    return df_augmented
