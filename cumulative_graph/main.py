import os

import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from cumulative_graph.detect_abrupt import detect_abrupt_changes
from cumulative_graph.detect_abrupt import detect_abrupt_changes_cusum
from cumulative_graph.manage_lines import least_squares_line
from cumulative_graph.manage_lines import get_line_equation
from cumulative_graph.analyze_results import calculate_aic
from cumulative_graph.analyze_results import calculate_error_metrics
from cumulative_graph.forecast_time_series import forecast_with_least_squares
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates



# Example usage
if __name__ == "__main__":

    value = '1'
    output_file_path = '../results/CV/2024/1_category/results.txt'
    output_df_file_path = '../results/CV/2024/1_category/df.csv'
    histogram_file_path = '../results/CV/2024/1_category/histogram.pdf'
    detect_changes_file_path = '../results/CV/2024/1_category/detect_changes.pdf'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Create a sample DataFrame with float64 time series data
    df = pd.read_csv('../sources/CV_2024.csv', header=None)
    df.columns = ['START_TIME','CATEGORY']
    df['START_TIME'] = pd.to_datetime(df['START_TIME'], format='%d-%m-%Y %H:%M:%S')
    df['CATEGORY'] = df['CATEGORY'].astype(str)
    print(df.head())
    print(len(df))
    print(df['START_TIME'].iloc[0])
    if value == 'all':
        print('Use Dataframe without filters\n')
    else:
        df.query("CATEGORY == @value", inplace=True)
        df = df.reset_index(drop=True)
        print("After filtering get a new DataFrame:\n")
        print(df.head())
        print(len(df))
    print(df['START_TIME'].iloc[0])
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    print(df.info())
    df['DELTA_TIME'] = df['START_TIME'] - df['START_TIME'].iloc[0]
    print(df.info())
    df['DELTA_SECONDS'] = df['DELTA_TIME'].dt.total_seconds() / 60
    print(df.head())
    print(df.info())
    df['TIME_DIFF'] = df['DELTA_SECONDS'].diff()
    print(df.head())
    print(df.info())
    df['INDEX'] = (1 / len(df)) * (df.index)
    print(df.head())
    print(df.tail())
    print(df.info())
    df.to_csv(output_df_file_path, index=True)

    df = df.dropna()
    # Define bin edges to match data
    bin_edges = np.arange(df['TIME_DIFF'].min(), df['TIME_DIFF'].max() + 1, 500)  # Step size of 100 sec
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['TIME_DIFF'], bins=bin_edges, edgecolor='black', alpha=0.7)
    # Remove empty space by adjusting x-axis limits
    plt.xlim(df['TIME_DIFF'].min(), df['TIME_DIFF'].max())
    # Labels and title
    plt.xlabel('Time Difference (seconds)')
    plt.ylabel('Frequency')
    plt.title('Histogram of TIME_DIFF')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Show plot
    plt.savefig(histogram_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Detect abrupt changes
    detected_changes = detect_abrupt_changes(df, start_event="START_TIME", time_column="DELTA_SECONDS", value_column="INDEX", model="rbf", pen=20)
    print("Detected Change Points with Coordinates:\n", detected_changes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["START_TIME"], df["INDEX"], label="Time Series")

    for _, row in detected_changes.iterrows():
        plt.axvline(row["START_TIME"], color="red", linestyle="--", label=f"Change Point at {row['START_TIME']}")
    plt.legend()
    plt.title("Abrupt Change Detection in Time Series")
    plt.xlabel("START_TIME")
    plt.ylabel("INDEX")
    plt.xlim(df['START_TIME'].iloc[0], df['START_TIME'].iloc[len(df)-1])
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(detect_changes_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # change_points_df = detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_SECONDS", threshold=40,
    #                                                drift=0.5)
    #
    # # Output detected change points as DataFrame
    # print("Change points detected:")
    # print(detected_changes)
    #
    # # Plot the time series with detected change points
    # plt.figure(figsize=(10, 6))
    # plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Time Series")
    # for _, row in change_points_df.iterrows():
    #     plt.axvline(row["DELTA_SECONDS"], color="red", linestyle="--", label=f"Change Point at {row['DELTA_SECONDS']:.2f}")
    # plt.legend()
    # plt.title("Time Series Change Point Detection (CUSUM)")
    # plt.xlabel("DELTA_SECONDS")
    # plt.ylabel("INDEX")
    # plt.xlim(0, df['DELTA_SECONDS'].iloc[len(df)-11])
    # plt.ylim(0, 1)
    # plt.grid()
    # plt.show()


    # Get the equations of the lines between change points using least squares
    with open(output_file_path, "w") as f:
        line_equations = get_line_equation(detected_changes, df)
        f.write("\nEquations of the best fit lines between change points (Least Squares):\n")
        for i, (m, b) in enumerate(line_equations):
            f.write(f"Line {i + 1}: y = {m:.8f}x + {b:.2f}\n")
        if not detected_changes.empty:
            last_change_point = detected_changes["DELTA_SECONDS"].iloc[-1]
        else:
            last_change_point = df["DELTA_SECONDS"].iloc[-1]
        f.write(f"Last change point: {last_change_point}")
    f = open(output_file_path, "r")
    print(f.read())
    # line_equations = get_line_equation(detected_changes, df)
    # print("\nEquations of the best fit lines between change points (Least Squares):")
    # for i, (m, b) in enumerate(line_equations):
    #     print(f"Line {i+1}: y = {m:.8f}x + {b:.2f}")

    # Get the last change point
    # if not detected_changes.empty:
    #     last_change_point = detected_changes["DELTA_SECONDS"].iloc[-1]
    # else:
    #     last_change_point = df["DELTA_SECONDS"].iloc[-1]
    #
    # print(f"Last change point: {last_change_point}")

    # Fit Least Squares (Linear Regression) model
    filtered_df = df[df["DELTA_SECONDS"] >= last_change_point].copy()
    X = filtered_df["DELTA_SECONDS"].values.reshape(-1, 1)
    y = filtered_df["INDEX"].values

    model = LinearRegression()
    model.fit(X, y)

    # Calculate AIC for the model
    aic = calculate_aic(model, X, y)
    print(f"Akaike's Information Criterion (AIC): {aic}")

    # Forecast future values using Least Squares (Linear Regression)
    forecast_period = 50000
    forecast_df = forecast_with_least_squares(df.copy(), value_column="INDEX", forecast_period=forecast_period,
                                              last_change_point=last_change_point)

    # Evaluate metrics on existing data
    y_true = filtered_df["INDEX"].values
    y_pred = model.predict(filtered_df["DELTA_SECONDS"].values.reshape(-1, 1))
    metrics = calculate_error_metrics(y_true, y_pred)

    print("\nError Metrics for the Model:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.8f}")

    print("\nForecasted Values using Least Squares:")
    print(forecast_df)

    # Plot the original time series, change points, and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Original Time Series")
    plt.scatter(detected_changes["DELTA_SECONDS"], detected_changes["INDEX"], color="red", label="Change Points")
    plt.plot(forecast_df["DELTA_SECONDS"], forecast_df["FORECAST"], label="Least Squares Forecast", linestyle="--")
    plt.legend()
    plt.title("Time Series with Detected Change Points and Forecast (Least Squares)")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.xlim(0)
    plt.ylim(0)
    plt.grid()
    plt.show()
