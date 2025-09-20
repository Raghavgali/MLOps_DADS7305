from sklearn.linear_model import LinearRegression
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Linear Regression regressor and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    joblib.dump(lin, "../Lab1_assignment/model/cali_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)