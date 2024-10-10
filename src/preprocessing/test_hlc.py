# pytest -s src/preprocessing/test_hlc.py

import pandas as pd
import pytest
from src.preprocessing.hlc import augment_with_hlc
from datetime import datetime

def get_hlc_data():
    august = datetime(2024, 8, 1) #thursday
    september = datetime(2024, 9, 2) #monday
    september_2nd_week_1 = datetime(2024, 9, 10) #tuesday : day, week, month
    september_2nd_week_2 = datetime(2024, 9, 11) #wednesday : day, week, month
    september_2nd_week_3 = datetime(2024, 9, 12) #thursday : day, week, month
    september_2nd_week_4 = datetime(2024, 9, 13) #friday : day, week, month


    data = {
        'time': [august, september, september_2nd_week_1, september_2nd_week_2, september_2nd_week_3, september_2nd_week_4],
        'open': [120, 122, 132, 142, 152, 162],
        'high': [125, 135, 145, 155, 165, 175],
        'low': [115, 120, 130, 140, 150, 160],
        'close': [122, 132, 142, 152, 162, 172]
    }

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df;


def test_augment_with_hlc():
    df = get_hlc_data()
    df_augmented = augment_with_hlc(df)
    df_augmented.to_csv("src/preprocessing/augmented_data.csv")

    expected_columns = [
        "open",
        "high",
        "low",
        "close",
        "prev_day_high",
        "prev_day_low",
        "prev_day_close",
        "prev_week_high",
        "prev_week_low",
        "prev_week_close",
        "prev_month_high",
        "prev_month_low",
        "prev_month_close",
    ]

    assert all(
        column in df_augmented.columns for column in expected_columns
    ), "Not all expected columns are present in the augmented DataFrame"

    # Check if the values in the augmented DataFrame are as expected
    assert df_augmented.loc["2024-09-11", "prev_day_high"] == 145
    assert df_augmented.loc["2024-09-11", "prev_day_low"] == 130
    assert df_augmented.loc["2024-09-11", "prev_day_close"] == 142

    assert df_augmented.loc["2024-09-11", "prev_week_high"] == 135
    assert df_augmented.loc["2024-09-11", "prev_week_low"] == 120
    assert df_augmented.loc["2024-09-11", "prev_week_close"] == 132

    assert df_augmented.loc["2024-09-11", "prev_month_high"] == 125
    assert df_augmented.loc["2024-09-11", "prev_month_low"] == 115
    assert df_augmented.loc["2024-09-11", "prev_month_close"] == 122

    rms_multiplier_d_high = 14.291946785935751
    assert df_augmented.loc["2024-09-11", "change_prev_day_high"] == 10 / 145 * rms_multiplier_d_high
    rms_multiplier_d_low = 14.34484067347489
    assert df_augmented.loc["2024-09-11", "change_prev_day_low"] == 10 / 130 * rms_multiplier_d_low
    rms_multiplier_d_close = 13.987500470527399
    assert df_augmented.loc["2024-09-11", "change_prev_day_close"] == 10 / 142 * rms_multiplier_d_close

    rms_multiplier_w_high = 5.4072375646914494
    assert df_augmented.loc["2024-09-11", "change_prev_week_high"] == 20 / 135 * rms_multiplier_w_high
    rms_multiplier_w_low = 4.876903611388027
    assert df_augmented.loc["2024-09-11", "change_prev_week_low"] == 20 / 120 * rms_multiplier_w_low
    rms_multiplier_w_close = 5.286716022525671
    assert df_augmented.loc["2024-09-11", "change_prev_week_close"] == 20 / 132 * rms_multiplier_w_close

    rms_multiplier_m_high = 3.768891807222045
    assert df_augmented.loc["2024-09-11", "change_prev_month_high"] == 30 / 125 * rms_multiplier_m_high
    rms_multiplier_m_low = 4.00378608698105
    assert df_augmented.loc["2024-09-11", "change_prev_month_low"] == 25 / 115 * rms_multiplier_m_low
    rms_multiplier_m_close = 3.6784384038487166
    assert df_augmented.loc["2024-09-11", "change_prev_month_close"] == 30 / 122 * rms_multiplier_m_close


# Run the test
if __name__ == "__main__":
    pytest.main()
