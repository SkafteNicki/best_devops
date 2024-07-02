from main import modulo

def test_modulo():
    assert modulo(4, 2) == 0
    assert modulo(5, 2) == 1
    assert modulo(6, 2) == 0