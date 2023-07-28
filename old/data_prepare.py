import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

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
    # Drop the 'Type of property' column
    data.drop(['Type of property'], axis=1, inplace=True)

    # Handle missing values using SimpleImputer with strategy 'constant'
    missing_value_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
    data = pd.DataFrame(missing_value_imputer.fit_transform(data), columns=data.columns)

    # Perform one-hot encoding on the 'Subtype of property' column
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    subtype_encoded = encoder.fit_transform(data[['Subtype of property']])

    # Get the feature names after one-hot encoding
    num_subtypes = len(encoder.categories_[0])
    column_names = ['Subtype_{}'.format(i) for i in range(num_subtypes)]
    subtype_encoded_df = pd.DataFrame(subtype_encoded, columns=column_names)

    # Drop the original 'Subtype of property' column
    data.drop('Subtype of property', axis=1, inplace=True)

    # Concatenate the one-hot encoded DataFrame with the original 'data' DataFrame
    data = pd.concat([data, subtype_encoded_df], axis=1)

    # Convert the entire DataFrame to integer type (Note: non-finite values will be handled)
    data = data.astype(int, errors='ignore')

    # save encoder
    joblib.dump(encoder, "oh_encoder.joblib")

    return data

