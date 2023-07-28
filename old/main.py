from data_import import find_csv_file, data_import
from data_import import data_import
from data_prepare import data_clean, data_preprocess
from data_format import train_test_split_data, scale_data
from model_select_train_score import model_select_train_score
from model_eval import model_error_calculate, model_crossvalidate
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib

def main():
    """
    Main function for running the regression model training, evaluation, and cross-validation.

    This function performs the following steps:
    1. Searches for a .csv file in the specified directory.
    2. If a .csv file is found, it imports the data into a DataFrame.
    3. Cleans and preprocesses the data.
    4. Splits the data into training and testing sets.
    5. Scales the data.
    6. Trains and evaluates the Linear Regression model.
    7. Trains and evaluates the XGBoost model.
    8. Displays the R2 scores for both models.
    9. Evaluates the models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
    10. Performs cross-validation on both models using 5-fold cross-validation and calculates the mean and standard deviation of RMSE scores.

    Note: The directory_path variable defines the directory where the CSV files are located. The function uses the find_csv_file
    function to search for a .csv file in that directory and read the data from it. If a .csv file is not found, an error message is displayed.

    Returns:
    None
    """
     
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

    # file name, using *.joblib as a file extension
    filename_xgboost = "xgboost.joblib"
    filename_linear_regression = "linear_regression.joblib"

    # save model
    joblib.dump(xgboost_model, filename_xgboost, compress = 3)
    joblib.dump(linear_regression_model, filename_linear_regression, compress = 3)

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
