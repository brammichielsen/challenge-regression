from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

def model_select_train_score(model, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the specified regression model using the provided training and testing data.

    Parameters:
    model (object): The regression model to be trained and evaluated.
    X_train (DataFrame): The feature matrix of the training data.
    X_test (DataFrame): The feature matrix of the testing data.
    y_train (Series): The target variable of the training data.
    y_test (Series): The target variable of the testing data.

    Returns:
    tuple: A tuple containing the trained model and the R2 score on the test data.
    """
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate R2 score on the test data
    r2_score = model.score(X_test, y_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Perform cross-validation (with num_folds deciding n-fold cross-validation)
    num_folds = 5
    scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')

    # Convert the scores from negative MSE to positive RMSE
    rmse_scores = -scores

    # Calculate the mean and standard deviation of RMSE scores
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    # Return the model and the R2 score
    return model, r2_score
