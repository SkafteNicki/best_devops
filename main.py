# Import necessary libraries
from typing import Tuple
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import argparse
import logging

def train_and_evaluate(test_size: float, kernel: str, random_state: int) -> Tuple[float, str, list]:
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=random_state)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)

    # Print cross-validation results
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

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