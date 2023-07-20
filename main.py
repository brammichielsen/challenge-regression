from src.data_import import find_csv_file, data_import
from src.data_import import data_import
from src.data_prepare import data_clean, data_preprocess
from src.data_format import train_test_split_data, scale_data
from src.model_select_train_score import model_select_train_score
from src.model_eval import model_error_calculate, model_crossvalidate
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def main():
    # Define the directory path where the CSV files are located
    directory_path = "data/"

    # Find a .csv file in the specified directory
    file_path = find_csv_file(directory_path)

    if file_path:
        # Read the dataset from the .csv file
        imported_data = data_import(file_path)
        # Now you can work with the imported_data DataFrame
    else:
        print("Error: Could not find a .csv file in the specified directory.")

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

    # Train and evaluate the Linear Regression model
    linear_regression_model = LinearRegression()
    linear_regression_model, linear_regression_score = model_select_train_score(linear_regression_model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Train and evaluate the XGBoost model
    xgboost_model = XGBRegressor()
    xgboost_model, xgboost_score = model_select_train_score(xgboost_model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Show the model scores
    print("Linear Regression Score:", linear_regression_score)
    print("XGBoost Score:", xgboost_score)
    print()

    # Evaluate the Linear Regression model using model_error_calculate function
    linear_regression_evaluation_results = model_error_calculate(y_test, linear_regression_model.predict(X_test_scaled))
    print("Linear Regression Model Evaluation Results:")
    print(linear_regression_evaluation_results)

    # Evaluate the XGBoost model using model_error_calculate function
    xgboost_evaluation_results = model_error_calculate(y_test, xgboost_model.predict(X_test_scaled))
    print("XGBoost Model Evaluation Results:")
    print(xgboost_evaluation_results)
    print()

    # Perform cross-validation using model_crossvalidate function for Linear Regression
    linear_regression_cv_results = model_crossvalidate(linear_regression_model, X_train_scaled, y_train, num_folds=5)
    print("Linear Regression Cross-Validation Results:")
    print(linear_regression_cv_results)

    # Perform cross-validation using model_crossvalidate function for XGBoost
    xgboost_cv_results = model_crossvalidate(xgboost_model, X_train_scaled, y_train, num_folds=5)
    print("XGBoost Cross-Validation Results:")
    print(xgboost_cv_results)

if __name__ == "__main__":
    main()
