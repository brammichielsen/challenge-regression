# DATA CLEANING

    # take dataset that was previously scraped, preprocessed and analyzed

    # handle NaNs

    # no duplicates

    # handle categorical (and numerical) data

    # no text data

    # select features and preprocess them as needed

    # remove features that have too strong correlation between them

# DATA FORMATTING

    #divide dataset for training and testing. (X_train, y_train, X_test, y_test)

    #if needed, apply scaling to your training data

# MODEL SELECTION

    #as a starting point, use the simple but powerful linear regression model

    #once your pipeline is fully ready, explore at least one more regression model

# TRAIN YOUR MODEL

    # train your model on your data

# MODEL EVALUATION

    # Evaluate your model with an appropriate metric

# MODEL EXPLAINABILITY

    # ?

from src.data_import import data_import
from src.data_clean import data_clean
from src.data_format import data_format
from src.model_select import model_select
from src.model_train import model_train
from src.model_eval import model_eval

def main():
    # Specify the file path of the CSV file
    file_path = "data/property_data.csv"

    # Call the data_import function to read the CSV file
    imported_data = data_import(file_path)

    # Clean data
    cleaned_data = data_clean(imported_data)

    # Format data
    formatted_data = data_format(cleaned_data)

    # Select model
    selected_model = model_select(formatted_data)

    # Train model
    trained_model = model_train(selected_model, formatted_data)

    # Evaluate the model
    model_eval(trained_model, formatted_data)

if __name__ == "__main__":
    main()
