from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

def model_error_calculate(y_test, y_pred):
    """
    Calculate Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

    Parameters:
    y_test (array-like): The true target values.
    y_pred (array-like): The predicted target values.

    Returns:
    dict: A dictionary containing the calculated error metrics.
    """

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    return {'Mean Squared Error': int(mse), 'Root Mean Squared Error': int(rmse), 'Mean Absolute Error': int(mae)}

def model_crossvalidate(model, X_train, y_train, num_folds=5):
    """
    Perform cross-validation on the specified model.

    Parameters:
    model (object): The regression model to be cross-validated.
    X_train (array-like): The feature matrix of the training data.
    y_train (array-like): The target variable of the training data.
    num_folds (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
    dict: A dictionary containing the mean and standard deviation of RMSE scores from cross-validation.
    """
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')

    # Convert the scores from negative MSE to positive RMSE
    rmse_scores = -scores

    # Calculate the mean and standard deviation of RMSE scores
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    return {'Mean RMSE': int(mean_rmse), 'Standard Deviation of RMSE': int(std_rmse)}
