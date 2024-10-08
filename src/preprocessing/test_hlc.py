import pandas as pd
import pytest
from hlc import augment_with_hlc
import matplotlib.pyplot as plt
from src.visualizations.line_chart import visualize_line_chart

df = pd.read_csv("src/data/SP_SPX_1D.csv")
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
print(df.tail())


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
    assert df_augmented.loc["2024-06-10", "prev_day_high"] == 5375.08
    assert df_augmented.loc["2024-06-10", "prev_day_low"] == 5331.33
    assert df_augmented.loc["2024-06-10", "prev_day_close"] == 5346.98

    assert df_augmented.loc["2024-06-10", "prev_week_high"] == 5375.08
    assert df_augmented.loc["2024-06-10", "prev_week_low"] == 5234.32
    assert df_augmented.loc["2024-06-10", "prev_week_close"] == 5346.98

    assert df_augmented.loc["2024-06-10", "prev_month_high"] == 5341.88
    assert df_augmented.loc["2024-06-10", "prev_month_low"] == 5011.05
    assert df_augmented.loc["2024-06-10", "prev_month_close"] == 5277.5


# Run the test
if __name__ == "__main__":
    pytest.main()
