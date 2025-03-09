import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_with_exponential_smoothing(df, value_column, forecast_period, last_change_point):
    """
    Forecast future values using Exponential Smoothing starting from the last change point.
    Args:
        df (pd.DataFrame): DataFrame with time series data.
        value_column (str): Column to forecast.
        forecast_period (int): Number of periods to forecast.
        last_change_point (float): The time corresponding to the last change point.

    Returns:
        pd.DataFrame: DataFrame with forecasted values.
    """
    # Filter data starting from the last change point
    filtered_df = df[df["DELTA_SECONDS"] >= last_change_point].copy()

    # Reset index for Exponential Smoothing
    filtered_df.reset_index(drop=True, inplace=True)

    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(
        filtered_df[value_column],
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    model_fit = model.fit()

    # Generate forecasts
    forecast_values = model_fit.forecast(forecast_period)

    # Create future timestamps
    last_timestamp = df[df["DELTA_SECONDS"] >= last_change_point]["DELTA_SECONDS"].iloc[-1]
    future_timestamps = [last_timestamp + i + 1 for i in range(forecast_period)]

    # Create a DataFrame for forecasts
    forecast_df = pd.DataFrame({
        "DELTA_SECONDS": future_timestamps,
        "FORECAST": forecast_values,
    })
    return forecast_df


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "DELTA_SECONDS": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "INDEX": [1.0, 1.1, 1.2, 5.0, 5.00001, 5.00002, 2.0, 2.1, 2.2, 8.0, 8.1, 8.2],
    }
    df = pd.DataFrame(data)

    # Detect changes using CUSUM
    change_points_df = detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_SECONDS", threshold=5,
                                                   drift=0.5)

    # Get the last change point
    if not change_points_df.empty:
        last_change_point = change_points_df["DELTA_SECONDS"].iloc[-1]
    else:
        last_change_point = df["DELTA_SECONDS"].iloc[0]

    print(f"Last change point: {last_change_point}")

    # Forecast future values using Exponential Smoothing
    forecast_period = 5
    forecast_df = forecast_with_exponential_smoothing(df.copy(), value_column="INDEX", forecast_period=forecast_period,
                                                      last_change_point=last_change_point)

    print("\nForecasted Values:")
    print(forecast_df)

    # Plot the original time series, change points, and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Original Time Series")
    plt.scatter(change_points_df["DELTA_SECONDS"], change_points_df["INDEX"], color="red", label="Change Points")
    plt.plot(forecast_df["DELTA_SECONDS"], forecast_df["FORECAST"], label="Exponential Smoothing Forecast",
             linestyle="--")
    plt.legend()
    plt.title("Time Series with Detected Change Points and Forecast")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.show()