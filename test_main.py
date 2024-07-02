from main import add
from main import subtractInt
import numpy as np

def test_add():
    assert add(1, 2) == 3
    assert add(1.0, 2.0) == 3.0
    assert add(3, 4) == 7
    assert (add(np.array([1, 2]), np.array([3, 4])) == np.array([4, 6])).all()
    assert (add(np.array([1, 2]), 3) == np.array([4, 5])).all()

# TODO: add more tests
def test_substractInt():
    assert subtractInt(5,2) == 3
    assert subtractInt(2, 3) == -1
    assert subtractInt(0,0) == 0