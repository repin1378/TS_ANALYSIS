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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression


def forecast_with_least_squares(df, value_column, forecast_period, last_change_point):
    """
    Forecast future values using the Method of Least Squares (Linear Regression) starting from the last change point.
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

    # Ensure data is sorted by time
    filtered_df = filtered_df.sort_values(by="DELTA_SECONDS")

    # Prepare data for linear regression
    X = filtered_df["DELTA_SECONDS"].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
    y = filtered_df[value_column].values

    # Fit the linear regression model (least squares)
    model = LinearRegression()
    model.fit(X, y)

    # Generate forecasts
    future_x = np.array([filtered_df["DELTA_SECONDS"].iloc[-1] + i + 1 for i in range(forecast_period)]).reshape(-1, 1)
    forecast_values = model.predict(future_x)

    # Create a DataFrame for forecasts
    forecast_df = pd.DataFrame({
        "DELTA_SECONDS": future_x.flatten(),
        "FORECAST": forecast_values,
    })
    return forecast_df