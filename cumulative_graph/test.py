import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

def detect_abrupt_changes(
    dataframe: pd.DataFrame,
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
        "DELTA_SECONDS": dataframe[time_column].iloc[change_points[:-1]].values,  # Ignore the last point as it's the end
        "INDEX": dataframe[value_column].iloc[change_points[:-1]].values
    })
    return results

def detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_SECONDS", threshold=5, drift=0.5):
    """
    Detects abrupt changes in a time series using the CUSUM (Cumulative Sum) algorithm.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        value_column (str): The column name of the time series data. Default is "INDEX".
        time_column (str): The column name representing time intervals. Default is "DELTA_SECONDS".
        threshold (float): The threshold for detecting change points.
        drift (float): The drift (acceptable variation) in the time series.

    Returns:
        pd.DataFrame: A DataFrame with the detected change points (DELTA_SECONDS and INDEX).
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
    change_points_df = pd.DataFrame(change_points, columns=["DELTA_SECONDS", "INDEX"])

    # Return the DataFrame with detected change points
    return change_points_df


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


def get_line_equation(change_points_df):
    """
    Generate the equations of the best fit lines between consecutive change points using least squares.

    Args:
        change_points_df (pd.DataFrame): DataFrame with change points, including DELTA_SECONDS and INDEX.

    Returns:
        list: A list of equations in the form of (m, b) for each line segment between change points.
    """
    equations = []

    for i in range(1, len(change_points_df)):
        # Get the x (DELTA_SECONDS) and y (INDEX) values for the current segment
        x_vals = change_points_df.iloc[i - 1:i + 1]["DELTA_SECONDS"].values
        y_vals = change_points_df.iloc[i - 1:i + 1]["INDEX"].values

        # Calculate the best fit line using least squares
        m, b = least_squares_line(x_vals, y_vals)
        equations.append((m, b))

    return equations


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


def calculate_aic(model, X, y):
    """
    Calculate Akaike's Information Criterion (AIC) for a given model.
    Args:
        model: Fitted regression model (e.g., sklearn's LinearRegression).
        X (array-like): Independent variable(s) used to fit the model.
        y (array-like): Dependent variable (actual values).

    Returns:
        float: AIC value.
    """
    # Predict the values using the model
    y_pred = model.predict(X)

    # Calculate the residual sum of squares (RSS)
    residual_sum_of_squares = np.sum((y - y_pred) ** 2)

    # Calculate the number of parameters (k) in the model
    k = X.shape[1] + 1  # Number of coefficients + intercept

    # Calculate the number of observations
    n = len(y)

    # Calculate the log-likelihood
    # Assuming Gaussian errors, log-likelihood is proportional to RSS
    log_likelihood = -n / 2 * (np.log(2 * np.pi * residual_sum_of_squares / n) + 1)

    # Calculate AIC
    aic = 2 * k - 2 * log_likelihood
    return aic


def calculate_error_metrics(y_true, y_pred):
    """
    Calculate common error metrics for regression/forecasting models.
    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: A dictionary with MAPE, ME, MAE, MPE, RMSE values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Ensure no division by zero for MAPE and MPE
    nonzero_indices = y_true != 0
    y_true_nonzero = y_true[nonzero_indices]
    y_pred_nonzero = y_pred[nonzero_indices]

    # Metrics calculation
    me = np.mean(y_pred - y_true)  # Mean Error
    mae = np.mean(np.abs(y_pred - y_true))  # Mean Absolute Error
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100  # MAPE in percentage
    mpe = np.mean((y_true_nonzero - y_pred_nonzero) / y_true_nonzero) * 100  # MPE in percentage
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))  # Root Mean Squared Error

    return {
        "MAPE (%)": mape,
        "ME": me,
        "MAE": mae,
        "MPE (%)": mpe,
        "RMSE": rmse
    }

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame with float64 time series data
    df = pd.read_csv('../sources/data.csv', header=None)
    df.columns = ['START_TIME']
    df['START_TIME'] = pd.to_datetime(df['START_TIME'], format='%d-%m-%Y %H:%M:%S')
    print(df.head())
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    print(df.info())
    df['DELTA_TIME'] = df['START_TIME'] - df['START_TIME'].iloc[0]
    print(df.info())
    df['DELTA_SECONDS'] = df['DELTA_TIME'].dt.total_seconds() / 60
    print(df.head())
    print(df.info())
    df['INDEX'] = (1 / len(df)) * (df.index)
    print(df.head())
    print(df.tail())
    print(df.info())

    # Detect abrupt changes
    detected_changes = detect_abrupt_changes(df, time_column="DELTA_SECONDS", value_column="INDEX", model="rbf", pen=10)
    print("Detected Change Points with Coordinates:\n", detected_changes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Time Series")
    for _, row in detected_changes.iterrows():
        plt.axvline(row["DELTA_SECONDS"], color="red", linestyle="--", label=f"Change Point at {row['DELTA_SECONDS']:.2f}")
    plt.legend()
    plt.title("Abrupt Change Detection in Time Series")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.xlim(0, df['DELTA_SECONDS'].iloc[len(df)-11])
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

    change_points_df = detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_SECONDS", threshold=40,
                                                   drift=0.5)

    # Output detected change points as DataFrame
    print("Change points detected:")
    print(change_points_df)

    # Plot the time series with detected change points
    plt.figure(figsize=(10, 6))
    plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Time Series")
    for _, row in change_points_df.iterrows():
        plt.axvline(row["DELTA_SECONDS"], color="red", linestyle="--", label=f"Change Point at {row['DELTA_SECONDS']:.2f}")
    plt.legend()
    plt.title("Time Series Change Point Detection (CUSUM)")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.xlim(0, df['DELTA_SECONDS'].iloc[len(df)-11])
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


    # Get the equations of the lines between change points using least squares
    line_equations = get_line_equation(change_points_df)
    print("\nEquations of the best fit lines between change points (Least Squares):")
    for i, (m, b) in enumerate(line_equations):
        print(f"Line {i+1}: y = {m:.8f}x + {b:.2f}")

    # Get the last change point
    if not change_points_df.empty:
        last_change_point = change_points_df["DELTA_SECONDS"].iloc[-1]
    else:
        last_change_point = df["DELTA_SECONDS"].iloc[0]

    print(f"Last change point: {last_change_point}")

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
    plt.scatter(change_points_df["DELTA_SECONDS"], change_points_df["INDEX"], color="red", label="Change Points")
    plt.plot(forecast_df["DELTA_SECONDS"], forecast_df["FORECAST"], label="Least Squares Forecast", linestyle="--")
    plt.legend()
    plt.title("Time Series with Detected Change Points and Forecast (Least Squares)")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.xlim(0)
    plt.ylim(0)
    plt.grid()
    plt.show()
