import joblib

def predict_new_data(preprocessed_data):
        
    xgboost_model = joblib.load("models/xgboost.joblib")
    linear_regression_model = joblib.load("models/linear_regression.joblib")

    # debug
    print("Model feature names:", xgboost_model.get_booster().feature_names)
    
    xgboost_predicted_price = xgboost_model.predict(preprocessed_data)
    linear_regression_predicted_price = linear_regression_model.predict(preprocessed_data)
    
    # Create a dictionary containing both predictions
    predictions = {
        "xgboost": xgboost_predicted_price.tolist(),
        "linear_regression": linear_regression_predicted_price.tolist()
    }

    # Return the predictions and status code
    return {"predictions": predictions, "status_code": 200}