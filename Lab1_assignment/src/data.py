import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the california housing dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the california housingdataset.
        y (numpy.ndarray): The target values of the california housing dataset.
    """
    cali = fetch_california_housing()
    X = cali.data
    y = cali.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test