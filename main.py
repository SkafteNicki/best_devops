
# Import necessary libraries
from typing import Tuple

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import argparse
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import argparse

def add(a: int | float | np.ndarray, b: int | float | np.ndarray) -> int | float | np.ndarray:
    return a + b

print("SVM model")

def train_and_evaluate(test_size: float, kernel: str, random_state: int) -> Tuple[float, str, list]:
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    # TODO: consider using cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=8)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train a Support Vector Machine (SVM) model
    if model_type == 'svm':
        model = SVC(kernel=kernel, random_state=42)
        if tune_hyperparameters:
            param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
            model = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        if tune_hyperparameters:
            param_grid = {'n_estimators': [10, 50, 100], 'max_features': ['auto', 'sqrt', 'log2']}
            model = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=1)
    else:
        raise ValueError("Unsupported model type")
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("HELLO SATANAS")

    logging.info(f'Accuracy on test set: {accuracy:.2f}')
    logging.info('Classification Report:')
    logging.info(report)
    return accuracy, report, cv_scores

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename='svm_results.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate an SVM model.")
    
    # Add arguments
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test set (default: 0.2)")
    parser.add_argument("--kernel", type=str, default='linear', help="Kernel type for SVM (default: 'linear')")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting (default: 42)")

    # Parse the arguments
    args = parser.parse_args()

    # Train and evaluate the model with the provided arguments
    accuracy, report, cv_scores = train_and_evaluate(args.test_size, args.kernel, args.random_state)

    # Train and evaluate the model with the provided arguments
    accuracy, report, cv_scores = train_and_evaluate(args.test_size, args.kernel, args.random_state)

    # Выведите результаты в консоль
    print(f'Accuracy on test set: {accuracy:.2f}')
    print('Classification Report:')
    print(report)
    print(f'Accuracy: {accuracy:.3f}')
    print('Classification Report:')
    print('Hello')
    print('Hello, my name is Sandra.')
    print(report)
    return accuracy, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--use_cross_validation', action='store_true', help='Whether to use cross-validation.')
    parser.add_argument('--model_type', type=str, default='svm', help='Type of model to use (e.g., "svm" or "random_forest").')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type to be used in the SVM model.')
    parser.add_argument('--use_poly_features', action='store_true', help='Whether to use polynomial features.')
    parser.add_argument('--degree', type=int, default=2, help='Degree of polynomial features.')
    parser.add_argument('--tune_hyperparameters', action='store_true', help='Whether to perform hyperparameter tuning.')

    args = parser.parse_args()
    train_and_evaluate(test_size=args.test_size, use_cross_validation=args.use_cross_validation, model_type=args.model_type, kernel=args.kernel, use_poly_features=args.use_poly_features, degree=args.degree, tune_hyperparameters=args.tune_hyperparameters)
