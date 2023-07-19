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
