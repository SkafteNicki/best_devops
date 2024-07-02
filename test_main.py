from main import add
import numpy as np

def test_add():
    assert add(1, 2) == 3
    assert add(1.0, 2.0) == 3.0
    assert add(3, 4) == 7
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

# TODO: add more tests