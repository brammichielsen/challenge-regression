import pandas as pd

def data_clean(data):
    """
    Clean the input dataset by performing the following operations:
    1. Drop duplicate rows.
    2. Drop irrelevant columns.
    3. Drop rows with NaN values in essential columns.
    4. Fill NaN values with 0 in specified columns.
    5. Drop columns containing NaN values.

    Parameters:
    data (DataFrame): The input dataset to be cleaned.

    Returns:
    DataFrame: The cleaned dataset.
    """
    
    # Drop duplicate rows
    data.drop_duplicates(inplace=True)

    # Drop irrelevant columns
    data.drop(['Raw num:', 'URL', 'ID number', 'Type of Sale', 'Locality', 'Zip code'], axis=1, inplace=True)

    # Drop rows with NaN values in essential columns
    data.dropna(subset=['Price of property in euro', 'Number of bedrooms', 'Living area'], inplace=True)

    # Fill NaN values with 0 in specified columns
    clean_values = {'Kitchen': 0, 'Terrace': 0, 'Garden': 0, 'Swimming pool': 0}
    data.fillna(clean_values, inplace=True)

    # Drop columns containing NaN values
    data.dropna(axis='columns', inplace=True)

    return data

def data_preprocess(data):
    """
    Preprocess the input dataset by performing the following operations:
    1. Drop the 'Type of property' column.
    2. Perform one-hot encoding on the 'Subtype of property' column.
    3. Drop the original 'Subtype of property' column.
    4. Concatenate the one-hot encoded DataFrame with the original 'data' DataFrame.
    5. Convert the entire DataFrame to integer type.

    Parameters:
    data (DataFrame): The input dataset to be preprocessed.

    Returns:
    DataFrame: The preprocessed dataset.
    """

    # Drop the 'Type of property' column
    data.drop(['Type of property'], axis=1, inplace=True)

    # Perform one-hot encoding on the 'Subtype of property' column
    one_hot_encoding = pd.get_dummies(data['Subtype of property'], prefix='Subtype', dtype=int)

    # Drop the original 'Subtype of property' column
    data.drop('Subtype of property', axis=1, inplace=True)

    # Concatenate the one-hot encoded DataFrame with the original 'data' DataFrame
    data = pd.concat([data, one_hot_encoding], axis=1)

    # Convert the entire DataFrame to integer type
    data = data.astype(int)

    return data
