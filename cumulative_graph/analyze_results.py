import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from cumulative_graph.detect_abrupt import detect_abrupt_changes
from cumulative_graph.detect_abrupt import detect_abrupt_changes_cusum
from cumulative_graph.manage_lines import least_squares_line
from cumulative_graph.manage_lines import get_line_equation
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression



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