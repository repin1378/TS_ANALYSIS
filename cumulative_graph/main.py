# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

# reading csv file
df = pd.read_csv('../sources/data.csv', header=None)
df.columns = ['START_TIME']
df['START_TIME'] = pd.to_datetime(df['START_TIME'], format='%d-%m-%Y %H:%M:%S')
print(df.head())
df['START_TIME'] = pd.to_datetime(df['START_TIME'])
print(df.info())
df['DELTA_TIME'] = df['START_TIME'] - df['START_TIME'].iloc[0]
print(df.info())
df['DELTA_SECONDS'] = df['DELTA_TIME'].dt.total_seconds()/60
print(df.head())
print(df.info())
df['INDEX'] = (1/len(df))*(df.index)
print(df.head())
print(df.tail())
print(df.info())

# create graph
# plt.figure(figsize=(10, 5))
# plt.plot(df['DELTA_SECONDS'], df['INDEX'])
# plt.xlabel('t, мин') #Подпись для оси х
# plt.ylabel('НПС') #Подпись для оси y
# plt.title('График накопленного числа событий') #Название
# # Set the range of x-axis
# plt.xlim(0, df['DELTA_SECONDS'].iloc[len(df)-11])
# # Set the range of y-axis
# plt.ylim(0, 1)
# plt.grid()
# plt.show()

num_of_rows = len(df)
print(num_of_rows)


def detect_linear_trend_changes(
    dataframe: pd.DataFrame,
    time_column: str = "DELTA_SECONDS",
    value_column: str = "INDEX",
    pen: float = 10.0
) -> pd.DataFrame:
    """
    Detect abrupt trend changes in time series data using the 'linear' model.

    Parameters:
    dataframe (pd.DataFrame): The input time series data with float64 columns.
    time_column (str): Name of the column containing the time information (float64).
    value_column (str): Name of the column containing the time series values (float64).
    pen (float): Penalty value for the change point detection algorithm.

    Returns:
    pd.DataFrame: A DataFrame containing the detected change points with time and value coordinates.
    """
    # Extract the time series data and reshape to 2D
    time_series = dataframe[value_column].values.reshape(-1, 1)

    # Initialize the change point detection algorithm with the linear model
    algo = rpt.Pelt(model="linear").fit(time_series)

    # Detect change points
    change_points = algo.predict(pen=pen)

    # Extract time and value coordinates for change points
    results = pd.DataFrame({
        "Time": dataframe[time_column].iloc[change_points[:-1]].values,  # Ignore the last point (end of data)
        "Value": dataframe[value_column].iloc[change_points[:-1]].values
    })
    return results

    # Detect abrupt trend changes
    pen = 10.0  # Example penalty value

    detected_changes = detect_linear_trend_changes(df, time_column="DELTA_SECONDS", value_column="INDEX", pen=pen)
    print("Detected Change Points with Coordinates:\n", detected_changes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["DELTA_SECONDS"], df["INDEX"], label="Time Series")
    for _, row in detected_changes.iterrows():
        plt.axvline(row["Time"], color="red", linestyle="--", label=f"Change Point at {row['Time']:.2f}")
    plt.legend()
    plt.title("Abrupt Trend Changes in Time Series (Linear Model)")
    plt.xlabel("DELTA_SECONDS")
    plt.ylabel("INDEX")
    plt.show()