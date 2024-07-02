from main import subtractInt

# TODO: add more tests
def test_substractInt():
    assert subtractInt(5,2) == 3
    assert subtractInt(2, 3) == -1
    assert subtractInt(0,0) == 0