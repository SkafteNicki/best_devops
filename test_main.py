from main import add
import numpy as np


def test_add():
    # Test with integers
    assert add(1, 2) == 3
    assert add(3, 4) == 7

    # Test with numpy arrays
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

    # Test with mixed types (int and float)
    assert add(1, 2.0) == 3.0
    assert add(2.5, 3) == 5.5

    # Test with negative numbers
    assert add(-1, -2) == -3
    assert add(-1.0, -2.0) == -3.0
    assert (add(np.array([-1, -2]), np.array([-3, -4])) == np.array([-4, -6])).all()
    assert (add(np.array([-1, -2]), -3) == np.array([-4, -5])).all()

    # Test with zeros
    assert add(0, 0) == 0
    assert add(0.0, 0.0) == 0.0
    assert (add(np.array([0, 0]), np.array([0, 0])) == np.array([0, 0])).all()
    assert (add(np.array([0, 0]), 0) == np.array([0, 0])).all()

    # Test with large numbers
    assert add(1e10, 1e10) == 2e10
    assert add(1e10, 1.0) == 1e10 + 1.0
    assert (add(np.array([1e10, 2e10]), np.array([3e10, 4e10])) == np.array([4e10, 6e10])).all()
    assert (add(np.array([1e10, 2e10]), 1e10) == np.array([2e10, 3e10])).all()

    # Test with mixed numpy arrays and scalars
    assert (add(np.array([1, 2]), np.array([3.0, 4.0])) == np.array([4.0, 6.0])).all()
    assert (add(np.array([1.0, 2.0]), 3) == np.array([4.0, 5.0])).all()


# Run the tests
if __name__ == "__main__":
    test_add()
    print("All tests passed!")
