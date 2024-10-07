# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/visualizations/line_chart.py

import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessing.hlc import augment_with_hlc
def visualize_line_chart(data, columns):
    """
    Visualize the specified columns from the given DataFrame as a line chart.

    Parameters:
    data (pd.DataFrame): The input data.
    columns (list of str): The columns to visualize.

    Returns:
    None
    """
    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

    data[columns].plot(kind='line')
    plt.yscale('log')
    for line in plt.gca().lines:
        line.set_linewidth(1)
    plt.title(f'{", ".join(columns)} over Time')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(columns)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('src/data/SP_SPX_1D.csv')
    df = df.head(100)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_augmented = augment_with_hlc(df)
    # df_augmented.reset_index(drop=True, inplace=True)
    df_augmented.to_csv('src/data/SP_SPX_1D_augmented.csv', sep='\t')
    visualize_line_chart(df_augmented, ['close', 'prev_day_close', 'prev_week_close', 'prev_month_close'])
