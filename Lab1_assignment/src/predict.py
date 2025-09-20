import joblib

def predict_data(X):
    """
    Predict the house prices for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted house prices.
    """
    model = joblib.load("../model/cali_model.pkl")
    y_pred = model.predict(X)
    return y_pred