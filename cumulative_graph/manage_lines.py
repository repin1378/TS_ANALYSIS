import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from cumulative_graph.detect_abrupt import detect_abrupt_changes
from cumulative_graph.detect_abrupt import detect_abrupt_changes_cusum
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression


def least_squares_line(x, y):
    """
    Calculates the equation of the best fit line using the least squares method.

    Args:
        x (np.array): The x-values (time values).
        y (np.array): The y-values (data points).

    Returns:
        tuple: The slope (m) and intercept (b) of the best fit line.
    """
    n = len(x)
    # Calculate the sums for the formula
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Calculate the slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)

    b = (sum_y - m * sum_x) / n

    return m, b


def get_line_equation(change_points_df, df):
    """
    Generate the equations of the best fit lines between consecutive change points using least squares.

    Args:
        change_points_df (pd.DataFrame): DataFrame with change points, including DELTA_SECONDS and INDEX.

    Returns:
        list: A list of equations in the form of (m, b) for each line segment between change points.
    """
    first_element = pd.DataFrame({'DELTA_SECONDS': [0], 'INDEX': [0]})
    print(first_element)
    last_element = df.iloc[-1][['DELTA_SECONDS', 'INDEX']]
    change_points_df = pd.concat([first_element, change_points_df], ignore_index=True)
    change_points_df.loc[len(change_points_df)] = last_element
    print(change_points_df)
    equations = []

    for i in range(1, len(change_points_df)):
        # Get the x (DELTA_SECONDS) and y (INDEX) values for the current segment
        x_vals = change_points_df.iloc[i - 1:i + 1]["DELTA_SECONDS"].values
        y_vals = change_points_df.iloc[i - 1:i + 1]["INDEX"].values

        # Calculate the best fit line using least squares
        m, b = least_squares_line(x_vals, y_vals)
        equations.append((m, b))

    return equations