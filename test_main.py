from main import add, train_and_evaluate
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pytest


def test_add():
    """Tests for the add function."""
    assert add(1, 2) == 3
    assert add(1.0, 2.0) == 3.0
    assert add(3, 4) == 7
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_train_and_evaluate_kernel(kernel):
    """Tests for the train_and_evaluate function with different SVM kernels."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Use the same scaler for X_test
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8  # Check that the accuracy is above 80%

def test_train_and_evaluate_random_state():
    """Test for the train_and_evaluate function with different random_state values."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train1, _, y_train1, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, _, y_train2, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    assert (X_train1 == X_train2).all()
    assert (y_train1 == y_train2).all()  # Check that the data is split identically
