from main import add
import numpy as np
import time
from main import train_and_evaluate

def test_add():
    assert add(1, 2) == 3
    assert add(1.0, 2.0) == 3.0
    assert add(3, 4) == 7
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

    # Invalid inputs
    try:
        add("1", 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")
    
    try:
        add(np.array([1, 2]), "3")
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")
    
    try:
        add(np.array([1, 2]), np.array([1, 2, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    print("All add() tests passed.")

# TODO: add more tests
def performance_test_add():
    large_array = np.random.rand(1000000)
    start_time = time.time()
    result = add(large_array, large_array)
    end_time = time.time()
    print(f"Performance test for add() with large arrays took {end_time - start_time:.5f} seconds.")
    
    assert (result == large_array + large_array).all()


def test_train_and_evaluate():
    # Basic test for default SVM
    accuracy, report = train_and_evaluate(test_size=0.2, model_type='svm')
    assert accuracy > 0.9  # Assuming the model should achieve more than 90% accuracy on the breast cancer dataset
    
    # Test for Random Forest
    accuracy, report = train_and_evaluate(test_size=0.2, model_type='random_forest')
    assert accuracy > 0.9  # Same assumption for Random Forest
    
    # Test with hyperparameter tuning for SVM
    accuracy, report = train_and_evaluate(test_size=0.2, model_type='svm', tune_hyperparameters=True)
    assert accuracy > 0.9

    # Test with hyperparameter tuning for Random Forest
    accuracy, report = train_and_evaluate(test_size=0.2, model_type='random_forest', tune_hyperparameters=True)
    assert accuracy > 0.9

    print("All train_and_evaluate() tests passed.")