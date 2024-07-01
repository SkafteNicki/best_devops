
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def add(a: int | float | np.ndarray, b: int | float | np.ndarray) -> int | float | np.ndarray:
    return a + b

print("SVM model")

def train_and_evaluate():
    # TODO: add arguments and argument parsing for high-level configuration

    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    # TODO: consider using cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Standardize the features
    # TODO: consider better feature engineering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Support Vector Machine (SVM) model
    # TODO: consider using a different model
    # TODO: consider hyperparameter tuning
    model = SVC(kernel='linear', random_state=8)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    # TODO: consider using more evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("HELLO SATANAS")

    print(f'Accuracy: {accuracy:.3f}')
    print('Classification Report:')
    print('Hello')
    print('Hello, my name is Sandra.')
    print(report)
    return accuracy, report

if __name__ == "__main__":
    train_and_evaluate()

    print(add(1, 2))
    print(add(1.5, 2.2))
    print(add(np.array([1, 2]), np.array([3, 4])))
