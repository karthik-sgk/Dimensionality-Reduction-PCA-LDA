import numpy as np
import pandas as pd

def calculate_standard_deviation(X: pd.DataFrame)-> np.ndarray:
    """ 
    Calculate the standard deviation of each feature in the dataset X.
    Args:
        X (numpy.ndarray or pandas.DataFrame): Input data of shape (n_samples, n_features).
        Returns:
        numpy.ndarray: Standard deviation of each feature, shape (n_features,).
    """
    std_dev = np.sqrt(np.mean((X - np.mean(X, axis=0)) ** 2, axis=0))
    return std_dev

def calculate_class_means(X, y):
    """
    Calculate the mean vector for each class.
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray or list): Class labels of shape (n_samples,).
    Returns:
        dict: Dictionary mapping class label to mean vector (numpy.ndarray).
    """
    class_means = {}
    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        class_means[cls] = np.mean(X[mask],axis=0)
    return class_means

def calculate_covariance_matrix(X):
    """ Calculate the covariance matrix
    Args:
        X (numpy.ndarray or pandas.DataFrame): Input data of shape (n_samples, n_features).
        Returns:
        numpy.ndarray: Covariance matrix of shape (n_features, n_features)."""
    covariance_matrix = np.matmul(X.T,X)/(X.shape[0] - 1)
    return covariance_matrix

def compute_class_covariances(X, y):
    """
    Compute the covariance matrix for each class.
    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Class labels.
    Returns:
        dict: Dictionary mapping class label to covariance matrix.
    """
    covariances = {}
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        X_cls_centered = X_cls - X_cls.mean()
        covariances[cls] = X_cls_centered.T @ X_cls_centered 
    return covariances

#
def calculate_eigen_decompostion(dat_matrix):
    """ Calculate eigenvalues and eigenvectors
    Args:
        cov_matrix (numpy.ndarray): Covariance matrix of shape (n_features, n_features).
        Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Sorted eigenvalues in descending order, shape (n_features,).
            - numpy.ndarray: Corresponding eigenvectors, shape (n_features, n_features).
    """
    eigenvalues, eigenvectors = np.linalg.eig(dat_matrix)
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors 