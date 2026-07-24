from HimalayanTokenization import PyHimalayanTokenization
def test_import():
    tok = PyHimalayanTokenization(mode="LM")
    assert tok is not None