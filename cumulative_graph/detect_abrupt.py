import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

def detect_abrupt_changes(
    dataframe: pd.DataFrame,
    start_event: str = "START_TIME",
    time_column: str = "TIME",
    value_column: str = "VALUE",
    model: str = "l2",
    pen: float = 10.0
) -> pd.DataFrame:
    """
    Detect abrupt changes in time series data and provide their coordinates.

    Parameters:
    dataframe (pd.DataFrame): The input time series data with float64 columns.
    time_column (str): Name of the column containing the time information (float64).
    value_column (str): Name of the column containing the time series values (float64).
    model (str): The cost function model to use ('l1', 'l2', 'rbf', etc.).
    pen (float): Penalty value for the change point detection algorithm.

    Returns:
    pd.DataFrame: A DataFrame containing the detected change points with time and value coordinates.
    """
    # Extract the time series data
    time_series = dataframe[value_column].values.reshape(-1, 1)

    # Initialize the change point detection algorithm
    algo = rpt.Pelt(model=model).fit(time_series)

    # Detect change points
    change_points = algo.predict(pen=pen)

    # Extract time and value coordinates for change points
    results = pd.DataFrame({
        "DELTA_MINUTES": dataframe[time_column].iloc[change_points[:-1]].values,  # Ignore the last point as it's the end
        "INDEX": dataframe[value_column].iloc[change_points[:-1]].values,
        "START_TIME": dataframe[start_event].iloc[change_points[:-1]].values
    })
    return results

def detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_MINUTES", threshold=5, drift=0.5):
    """
    Detects abrupt changes in a time series using the CUSUM (Cumulative Sum) algorithm.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        value_column (str): The column name of the time series data. Default is "INDEX".
        time_column (str): The column name representing time intervals. Default is "DELTA_MINUTES".
        threshold (float): The threshold for detecting change points.
        drift (float): The drift (acceptable variation) in the time series.

    Returns:
        pd.DataFrame: A DataFrame with the detected change points (DELTA_MINUTES and INDEX).
    """
    # Extract the time series values
    time_series = df[value_column].values
    time_stamps = df[time_column].values

    # Initialize the CUSUM variables
    cusum_pos = np.zeros(len(time_series))  # CUSUM for positive deviations
    cusum_neg = np.zeros(len(time_series))  # CUSUM for negative deviations
    change_points = []

    # Loop through the time series and calculate the CUSUM
    for i in range(1, len(time_series)):
        # Calculate the positive and negative CUSUMs
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (time_series[i] - time_series[i - 1] - drift))
        cusum_neg[i] = min(0, cusum_neg[i - 1] + (time_series[i - 1] - time_series[i] - drift))

        # Check if the positive or negative CUSUM exceeds the threshold
        if cusum_pos[i] > threshold:
            change_points.append((time_stamps[i], time_series[i]))
            cusum_pos[i] = 0  # Reset after detecting a change
        elif cusum_neg[i] < -threshold:
            change_points.append((time_stamps[i], time_series[i]))
            cusum_neg[i] = 0  # Reset after detecting a change

    # Create a DataFrame for the detected change points
    change_points_df = pd.DataFrame(change_points, columns=["DELTA_MINUTES", "INDEX"])

    # Return the DataFrame with detected change points
    return change_points_df