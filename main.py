from src.data_import import data_import
from src.data_prepare import data_clean, data_preprocess
from src.data_format import train_test_split_data, scale_data
from model_select_train_score import model_select_train_score
from model_eval import model_error_calculate, model_crossvalidate

def main():
    # Specify the file path of the CSV file
    file_path = "data/property_data.csv"

    # Call the data_import function to read the CSV file
    imported_data = data_import(file_path)

    # Clean data
    cleaned_data = data_clean(imported_data)

    # Preprocess data
    preprocessed_data = data_preprocess(cleaned_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_data(preprocessed_data)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Train and evaluate the model
    trained_model, r2_score = model_select_train_score(X_train_scaled, X_test_scaled, y_train, y_test)

    # Evaluate the model using model_error_calculate function
    evaluation_results = model_error_calculate(y_test, trained_model.predict(X_test_scaled))
    print("Model Evaluation Results:")
    print(evaluation_results)

    # Perform cross-validation using model_crossvalidate function
    cv_results = model_crossvalidate(trained_model, X_train_scaled, y_train, num_folds=5)
    print("Cross-Validation Results:")
    print(cv_results)

if __name__ == "__main__":
    main()
