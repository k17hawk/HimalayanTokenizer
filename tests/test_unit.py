
import HimalayanTokenization
def test_import():
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    assert tok is not None