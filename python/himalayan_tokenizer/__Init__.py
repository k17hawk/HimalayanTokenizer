from importlib.resources import files
from HimalayanTokenization import PyHimalayanTokenization

def load_default_tokenizer(mode="LM"):
    tok = PyHimalayanTokenization(mode=mode)
    vocab_path = files("himalayan_tokenizer").joinpath("vocab.tsv")
    tok.load_vocab_tsv(str(vocab_path))
    return tok