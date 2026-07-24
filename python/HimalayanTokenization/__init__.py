from .HimalayanTokenization import PyHimalayanTokenization
from importlib.resources import files

def load_default_tokenizer(mode="LM"):
    tok = PyHimalayanTokenization(mode=mode)
    vocab_path = files(__name__).joinpath("vocab.tsv")
    tok.load_vocab_tsv(str(vocab_path))
    return tok