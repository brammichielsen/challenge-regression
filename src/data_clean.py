import pandas as pd

def data_clean(data):
    """
    Cleans the data by handling NaN values, duplicates, and data types.
    
    Parameters:
    data (DataFrame): The input data to be cleaned.
    
    Returns:
    DataFrame: The cleaned data.
    """
    # Handle NaN values
    data.dropna(inplace=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Convert categorical columns to categorical data type
    categorical_columns = ['column1', 'column2', ...]  # Specify the categorical column names
    data[categorical_columns] = data[categorical_columns].astype('category')

    # Convert numerical columns to numerical data type
    numerical_columns = ['column3', 'column4', ...]  # Specify the numerical column names
    data[numerical_columns] = data[numerical_columns].astype('float')

    # Return the cleaned data
    return data
