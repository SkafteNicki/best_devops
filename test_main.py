from main import add
import numpy as np

def test_add():
    assert add(1, 2) == 3
    assert add(1.0, 2.0) == 3.0
    assert add(3, 4) == 7
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

    # TODO: add more tests

    # Test with mixed types
    assert add(1, 2.0) == 3.0
    assert add(2.0, 1) == 3.0
    assert (add(np.array([1, 2]), 2.0) == np.array([3.0, 4.0])).all()
    assert (add(2.0, np.array([1, 2])) == np.array([3.0, 4.0])).all()

     # Test with multi-dimensional arrays
    assert (add(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])) == np.array([[6, 8], [10, 12]])).all()
    assert (add(np.array([[1, 2], [3, 4]]), 1) == np.array([[2, 3], [4, 5]])).all()
    



