#pytest -v src/preprocessing/test_hlc.py

import pandas as pd
import pytest
from hlc import augment_with_hlc
from datetime import datetime, timedelta

today = datetime(2024, 10, 9)
yesterday = today - timedelta(days=1)
last_week = today - timedelta(weeks=1)
last_month = today - timedelta(days=30)


data = {
    'time': [last_month, last_week, yesterday, today],  # Times for the four rows
    'open': [120, 130, 140, 150],
    'high': [125, 135, 145, 155],
    'low': [115, 125, 135, 145],
    'close': [122, 132, 142, 152]
}
df = pd.DataFrame(data)
df.set_index("time", inplace=True)


def test_augment_with_hlc():
    # Call the function to be tested
    df_augmented = augment_with_hlc(df)
    # Save the augmented DataFrame to a CSV file in the current folder
    df_augmented.to_csv("src/preprocessing/augmented_data.csv")

    # Check if the augmented DataFrame has the expected columns
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
    assert df_augmented.loc["2024-10-09", "prev_day_high"] == 145
    assert df_augmented.loc["2024-10-09", "prev_day_low"] == 135
    assert df_augmented.loc["2024-10-09", "prev_day_close"] == 142

    assert df_augmented.loc["2024-10-09", "prev_week_high"] == 135
    assert df_augmented.loc["2024-10-09", "prev_week_low"] == 125
    assert df_augmented.loc["2024-10-09", "prev_week_close"] == 132

    assert df_augmented.loc["2024-10-09", "prev_month_high"] == 125
    assert df_augmented.loc["2024-10-09", "prev_month_low"] == 115
    assert df_augmented.loc["2024-10-09", "prev_month_close"] == 122


    assert df_augmented.loc["2024-10-09", "change_prev_day_high"] == 10 / 145
    assert df_augmented.loc["2024-10-09", "change_prev_day_low"] == 10 / 135
    assert df_augmented.loc["2024-10-09", "change_prev_day_close"] == 10 / 142

    assert df_augmented.loc["2024-10-09", "change_prev_week_high"] == 20 / 135
    assert df_augmented.loc["2024-10-09", "change_prev_week_low"] == 20 / 125
    assert df_augmented.loc["2024-10-09", "change_prev_week_close"] == 20 / 132

    assert df_augmented.loc["2024-10-09", "change_prev_month_high"] == 30 / 125
    assert df_augmented.loc["2024-10-09", "change_prev_month_low"] == 30 / 115
    assert df_augmented.loc["2024-10-09", "change_prev_month_close"] == 30 / 122


# Run the test
if __name__ == "__main__":
    pytest.main()
