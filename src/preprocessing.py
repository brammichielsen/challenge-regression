import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

def preprocess_new_data(new_house_data):
    # Convert the dictionary to a DataFrame
    data = pd.DataFrame(new_house_data["data"], index=[0])

    # Load the saved encoder
    encoder = joblib.load("models/oh_encoder.joblib")

    # Ensure the column name matches the encoder's feature names
    data.rename(columns={"subtype_of_property": "Subtype of property"}, inplace=True)

    # Perform one-hot encoding on the 'Subtype of property' column using the loaded encoder
    subtype_encoded = encoder.transform(data[["Subtype of property"]])
    subtype_encoded_df = pd.DataFrame(subtype_encoded, columns=encoder.get_feature_names_out(["Subtype of property"]))

    # Drop the original 'Subtype of property' column
    data.drop(columns=["Subtype of property"], inplace=True)

    # Concatenate the one-hot encoded DataFrame with the original 'data' DataFrame
    data = pd.concat([data, subtype_encoded_df], axis=1)

    return data