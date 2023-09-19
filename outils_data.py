import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import svm
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2_contingency

def draw_boxplots(numeric_columns: list, df : pd.DataFrame):
    """
    Draw box plots for numeric columns.

    Args:
        numeric_columns (list): List of column names containing numeric data.
        df (pd.Dataframe) : Dataframe where are the numeric data

    Returns:
        None
    """

    # Create a figure and subplots for individual plots
    fig, axes = plt.subplots(nrows=1, ncols=len(numeric_columns), figsize=(18, 6))

    # Iterate through each feature and create a box plot using Seaborn
    for i, feature in enumerate(numeric_columns):
        sns.boxplot(data=df[feature], ax=axes[i], color='#3498db', width=0.5, linewidth=2)
        axes[i].set_title(f'{feature}', fontsize=10)  # Set subplot title
        axes[i].set_xlabel('')  # Clear x-axis label
        axes[i].set_ylabel(feature, fontsize=12)  # Set y-axis label
        axes[i].tick_params(axis='both', labelsize=12)  # Set tick label size for both axes

    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()

    # Set the main title for the entire plot
    plt.suptitle('Distribution of Selected Features', fontsize=16, y=1.02)

    # Display the plot
    plt.show()

def iqr(df, feature_column):
    """
    Calculate the Interquartile Range (IQR) for a specified column and identify anomalies.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feature_column (str): The column for which to calculate IQR and identify anomalies.

    Returns:
        pandas.DataFrame: DataFrame with an added column indicating anomalies.
    """
    Q1 = df[feature_column].quantile(0.25)  # Calculate the first quartile (Q1)
    Q3 = df[feature_column].quantile(0.75)  # Calculate the third quartile (Q3)
    IQR = Q3 - Q1  # Calculate the Interquartile Range (IQR)

    anomaly_col_name = "anomaly_iqr_" + str(feature_column)  # Generate the name for the anomaly column

    # Create a new column indicating anomalies based on IQR calculations
    df[anomaly_col_name] = ((df[feature_column] < (Q1 - 1.5 * IQR)) | (df[feature_column] > (Q3 + 1.5 * IQR)))

    return df

def zscore(df, feature_column):
    """
    Detect outliers in the specified column using the Z-score method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feature_column (str): The column to analyze for outliers.

    Returns:
        pandas.DataFrame: DataFrame with an added column indicating outliers.
    """
    mean = df[feature_column].mean()  # Calculate the mean of the column
    stddev = df[feature_column].std()  # Calculate the standard deviation of the column
    upper_limit = mean + (3 * stddev)  # Calculate the upper limit for outliers
    lower_limit = mean - (3 * stddev)  # Calculate the lower limit for outliers

    anomaly_col_name = "anomaly_zscore_" + str(feature_column)  # Generate the name for the anomaly column

    # Determine the outliers using the Z-score method
    df[anomaly_col_name] = ((df[feature_column] < lower_limit) | (df[feature_column] > upper_limit))

    return df

def isolation_forest(df, feature_column, cont):
    """
    Detect anomalies in the specified column using the Isolation Forest method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feature_column (str): The column to analyze for anomalies.
        cont (float): Contamination parameter for the Isolation Forest.

    Returns:
        pandas.DataFrame: DataFrame with an added column indicating anomalies detected by Isolation Forest.
    """
    # Create an Isolation Forest model
    isolation_forest = IsolationForest(contamination=cont)

    df_isolation = df.loc[:, [feature_column]]  # Select only the specified column for analysis

    # Train the Isolation Forest model on the data
    isolation_forest.fit(df_isolation)

    # Predict anomalies (-1 label for anomalies, 1 for normal data)
    predictions = isolation_forest.predict(df_isolation)

    # Add the predictions to the original DataFrame
    df_isolation['anomaly'] = predictions

    anomaly_col_name = "anomaly_isolationforest_" + str(feature_column)  # Generate the name for the anomaly column

    # Convert Isolation Forest predictions to True for anomalies and False for normal data
    df[anomaly_col_name] = df_isolation['anomaly'].apply(lambda x: True if x == -1 else False)

    return df

def oneclassSVM(df, feature_column, nu):
    """
    Detect anomalies in the specified feature column using the One-Class SVM method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feature_column (str): The column to analyze for anomalies.
        nu (float): The fraction of training errors and a lower bound of the fraction of support vectors.

    Returns:
        pandas.DataFrame: DataFrame with an added column indicating anomalies detected by One-Class SVM.
    """
    data = df[feature_column].values.reshape(-1, 1)  # Extract data from the specified column

    model = svm.OneClassSVM(nu=nu, kernel="rbf")  # Create a One-Class SVM model

    model.fit(data)  # Train the model on normal data

    predictions = model.predict(data)  # Predict whether column data is an anomaly

    anomaly_col_name = "anomaly_oneclassSVM_" + str(feature_column)  # Generate the name for the anomaly column

    # Add the predictions to the original DataFrame
    df[anomaly_col_name] = predictions

    # Convert One-Class SVM predictions to True for anomalies and False for normal data
    df[anomaly_col_name] = df[anomaly_col_name].apply(lambda x: True if x == -1 else False)

    return df

def analyze(df, feature_column, *args, cont=0.01, nu=0.1):
    """
    Analyze the input DataFrame for outliers using specified methods.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feature_column (str): The column to analyze for anomalies.
        *args: Variable number of anomaly detection methods.
        cont (float, optional): Contamination parameter for methods like Isolation Forest. Default is 0.01.
        nu (float, optional): The fraction of training errors for One-Class SVM. Default is 0.1.

    Returns:
        pandas.DataFrame: DataFrame containing the results of outlier detection methods.
    """
    output_df = pd.DataFrame()  # Initialize an empty DataFrame to store the results

    for arg in args:
        if (arg.__name__ == "zscore") or (arg.__name__ == "iqr"):
            output_df = arg(df, feature_column)  # Call the zscore or iqr function and store the results in output_df
        if (arg.__name__ == "isolation_forest") :
            output_df = arg(df, feature_column, cont)  # Call the isolation_forest function with contamination parameter
        if (arg.__name__ == "oneclassSVM") :
            output_df = arg(df, feature_column, nu)  # Call the oneclassSVM function with contamination parameter

    return output_df

def dataframe_outliers(df, columns, *args, cont=0.02, nu=0.1):
    """
    Detect and count outliers in a DataFrame for specified columns using various methods.

    Args:
        df (DataFrame): The input DataFrame to analyze.
        columns (list): A list of column names to analyze for outliers.
        *args: Additional arguments that can be passed to the 'analyze' function.
        cont (float): The contamination parameter for outlier detection (default is 0.02).
        nu (float): The nu parameter for outlier detection (default is 0.1).

    Returns:
        DataFrame: A new DataFrame containing the original data along with columns
                    indicating the number of outliers for each specified column.
    """

    # Extract column names from the input DataFrame
    list_columns = df.columns
    df_to_return = df[list_columns]

    for column in columns:
        # Analyze the DataFrame using different outlier detection methods
        df_for_work = analyze(df[list_columns], column, *args, cont=cont, nu=nu)

        # Identify columns with names starting with "anomaly"
        anomaly_columns = [col for col in df_for_work.columns if col.startswith('anomaly')]

        col_name = "anomaly_" + column

        # Create a new column to count the number of anomalies for the current column
        df_for_work[col_name] = 0
        for anomaly in anomaly_columns:
            df_for_work[col_name] += df_for_work[anomaly].astype(int)

        # Convert the anomaly count column to a list
        list_anomaly = df_for_work[col_name].to_list()

        # Add the new anomaly count column to the DataFrame to return
        df_to_return[col_name] = list_anomaly

    return df_to_return

def outliers_by_column(df_outliers, column, threshold):
    """
    Extract rows from a DataFrame that have outlier values in a specific column
    based on a given threshold.

    Args:
        df_outliers (DataFrame): The input DataFrame containing outlier information.
        column (str): The name of the column for which outliers are filtered.
        threshold (int or float): The threshold value to identify outliers in the column.

    Returns:
        DataFrame: A new DataFrame containing rows from the input DataFrame
                    where the specified column's value exceeds the given threshold.
    """

    # Create a copy of the input DataFrame for manipulation
    df_for_work = df_outliers

    # Create a column name for the anomaly count associated with the specified column
    col_name = "anomaly_" + column

    # Filter rows where the anomaly count for the specified column is greater than or equal to the threshold
    df_anomaly = df_for_work[df_for_work[col_name] >= threshold]

    # Create a list of columns to return, excluding the anomaly count columns
    cols_to_return = [col for col in df_for_work.columns if not (col.startswith('anomaly'))]

    # Include the anomaly count column for the specified column in the columns to return
    cols_to_return.append(col_name)

    # Create a new DataFrame containing only the selected columns
    df_to_return = df_anomaly[cols_to_return]

    return df_to_return

def distribution_curve(df, variable):
    """
    Plot the distribution curve for a specified variable in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        variable (str): The name of the variable to be analyzed within the DataFrame.
    """

    # Extract the variable to be analyzed from the DataFrame
    variable_to_analyze = df[variable]

    # Print descriptive statistics of the variable
    print(variable_to_analyze.describe())

    # Plot a histogram with distinct bar colors
    plt.figure(figsize=(8, 6))
    sns.histplot(variable_to_analyze, kde=True, color='b')
    plt.title(f'{variable} variable Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def normal_distribution(df, variable, alpha):
    """
    Perform the Shapiro-Wilk normality test on a specified variable within a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        variable (str): The name of the variable to be analyzed within the DataFrame.
        alpha (float): The significance level for the test.

    Returns:
        bool: True if the p-value of the Shapiro-Wilk test is greater than alpha, indicating normality.
    """

    # Perform the Shapiro-Wilk test
    statistic, p_value = stats.shapiro(df[variable])

    # Check if the p-value is greater than alpha
    return p_value > alpha

def draw_pieplot(df, column):
    """
    Generate a pie chart to visualize the distribution of unique values in a column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column to be visualized.

    Returns:
        None
    """

    # Count the occurrences of each unique value in the specified column
    counts = df[column].value_counts()

    # Generate random colors for each slice of the pie chart
    colors = np.random.rand(len(counts), 3)

    # Create a new figure for the pie chart
    plt.figure(figsize=(8, 8))

    # Generate the pie chart with labels, percentages, and starting angle
    _, _, text = plt.pie(counts, labels=counts.index, autopct='%1.2f%%', startangle=90, colors=colors)

    # Change the color of texts
    for t in text:
        t.set_color('black')

    # Set the title of the pie chart
    plt.title(f'Distribution of the variable {column}')

    # Make the pie chart circular by setting the aspect ratio to be equal
    plt.axis('equal')

    # Display the pie chart
    plt.show()

def draw_barplot(df, column):
    """
    Generate a bar plot to visualize the frequency of unique values in a column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column to be visualized.

    Returns:
        None
    """

    # Count the occurrences of each unique value in the specified column
    counts = df[column].value_counts()

    # Define colors for the bars
    colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral']

    # Create a new figure for the bar plot
    plt.figure(figsize=(8, 6))

    # Generate the bar plot
    counts.plot(kind='bar', color=colors)

    # Set the title, x-axis label, and y-axis label
    plt.title(f'Number of Occurrences for each value of {column}')
    plt.xlabel(column)
    plt.ylabel('Number of Occurrences')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=0)

    # Add text labels on top of each bar
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=10, color='black')

    # Display the bar plot
    plt.show()

def chi_squared_tests(df, categorical_columns):
    """
    Perform χ² tests between pairs of categorical variables.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the categorical variables.
        categorical_columns (list): List of column names containing categorical variables.

    Returns:
        None (prints test results).
    """

    # Generate unique pairs of categorical variables
    pairs_to_test = []
    for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            pairs_to_test.append((categorical_columns[i], categorical_columns[j]))

    # Perform χ² test for each pair of variables
    for pair in pairs_to_test:
        contingency_table = pd.crosstab(df[pair[0]], df[pair[1]])
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

        print(f"Test du χ² entre {pair[0]} et {pair[1]}:")
        print(f"Statistique de test du χ² : {chi2_stat}")
        print(f"Valeur de p : {p_val}")
        print(f"Degrés de liberté : {dof}")
        print("Fréquences attendues :")
        print(expected)
        print("\n")
