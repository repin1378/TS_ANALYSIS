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
        change_points_df (pd.DataFrame): DataFrame with change points, including DELTA_MINUTES and INDEX.

    Returns:
        list: A list of equations in the form of (m, b) for each line segment between change points.
    """
    first_element = pd.DataFrame({'DELTA_MINUTES': [0], 'INDEX': [0], 'START_TIME': df.iloc[0][['START_TIME']]})
    last_element = df.iloc[-1][['DELTA_MINUTES', 'INDEX', 'START_TIME']]
    change_points_df = pd.concat([first_element, change_points_df], ignore_index=True)
    change_points_df.loc[len(change_points_df)] = last_element
    print(change_points_df)
    equations = []

    for i in range(1, len(change_points_df)):
        # Get the x (DELTA_MINUTES) and y (INDEX) values for the current segment
        x_vals = change_points_df.iloc[i - 1:i + 1]["DELTA_MINUTES"].values
        y_vals = change_points_df.iloc[i - 1:i + 1]["INDEX"].values

        # Calculate the best fit line using least squares
        m, b = least_squares_line(x_vals, y_vals)
        equations.append((m, b))

    return equations

#Функция для создания отчета
def create_report(df, change_points_df, line_equations):

    first_element = pd.DataFrame({
        'DELTA_MINUTES': [0],
        'INDEX': [0],
        'START_TIME': [df.iloc[1]['START_TIME']]
    })
    last_element = df.iloc[-1][['DELTA_MINUTES', 'INDEX', 'START_TIME']]
    change_points_df = pd.concat([first_element, change_points_df], ignore_index=True)
    change_points_df.loc[len(change_points_df)] = last_element
    print('Change Point Frame:\n')
    print(change_points_df)
    print(change_points_df.dtypes)

    change_points_df['START_TIME'] = pd.to_datetime(df['START_TIME'], format='%d-%m-%Y %H:%M:%S')
    #change_points_df['START_TIME'] = change_points_df['START_TIME'].dt.strftime('%d-%m-%Y %H:%M')
    print(change_points_df.iloc[0])

    df_report = pd.DataFrame(columns=['Seconds_start_point','Date_start_point','Seconds_last_point','Date_last_point','Equations_of_line','Monthly_index'])
    for i, (m, b) in enumerate(line_equations):
        str_line=f"Line {i + 1}: y = {m:.8f}x + ({b:.8f})"
        monthly_index = m*43800
        new_row = {'Seconds_start_point': change_points_df["DELTA_MINUTES"].iloc[i], 'Date_start_point': change_points_df["START_TIME"].iloc[i], 'Seconds_last_point': change_points_df["DELTA_MINUTES"].iloc[i+1], 'Date_last_point': change_points_df["START_TIME"].iloc[i+1],'Equations_of_line': str_line,'Monthly_index': round(monthly_index,5)}
        df_report.loc[i] = new_row
    print(df_report)

    return df_report