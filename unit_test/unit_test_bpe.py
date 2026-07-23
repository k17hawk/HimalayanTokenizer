"""
Unit tests for NepBPE v2 tokenizer invariants.
These tests verify the exact guarantees claimed in the equation set.
"""
import pytest
import unicodedata
from itertools import chain
from typing import Set

from NPBPE_tokenizer import NepBPETokenizer, normalize, script_of, AksharaTokenizer, FOLD_TABLE


def test_normalisation_idempotent():
    """N(s) = N(N(s)) for various inputs."""
    cases = [
        "सँग",           # already canonical
        "संग",           # must be folded
        "hello world",
        "सँग hello संग",
        "\u200D\u200C",  # ZWJ+ZWNJ
    ]
    for s in cases:
        n1 = normalize(s)
        n2 = normalize(n1)
        assert n1 == n2, f"Idempotency failed for {repr(s)}"


def test_folding_table_reduces_variants():
    """All variant forms should map to the same canonical string."""
    variants = ["सँग", "संग", "सङ्ग"]
    canonical = normalize("सँग")
    for v in variants:
        assert normalize(v) == canonical, f"Variant {repr(v)} not folded"


def test_zwnj_preserved_zwj_stripped():
    s = "\u200D\u200C"
    n = normalize(s)
    assert "\u200D" not in n, "ZWJ should be stripped"
    assert "\u200C" in n, "ZWNJ should be preserved"


# -------------------------------------------------------------------
# 2. Akshara integrity (no split matra, no broken conjuncts)
# -------------------------------------------------------------------
def is_well_formed_akshara(token: str) -> bool:
    """Check that a Devanagari token forms a single akshara.
    
    A well-formed akshara:
    - May consist of a consonant cluster optionally followed by a vowel sign
    - May end with a virama (halanta) indicating suppressed inherent vowel
    - Must not have a virama followed by anything other than a consonant or end-of-token
    """
    if not any(0x0900 <= ord(c) <= 0x097F for c in token):
        return True
    
    virama = "\u094D"
    if virama not in token:
        return True  # no virama = no problem
    
    # Check each virama position
    for i, ch in enumerate(token):
        if ch == virama:
            # Virama at the very end is valid (halanta form)
            if i + 1 >= len(token):
                continue
            # Virama must be followed by a consonant (conjunct formation)
            next_cp = ord(token[i+1])
            if not (0x0915 <= next_cp <= 0x0939):
                return False
    
    return True


def test_akshara_integrity_on_devanagari():
    tokenizer = AksharaTokenizer()
    test_strings = [
        "मेरो",
        "स्कुल",        # conjunct स् + कु + ल  = स्कुल
        "विद्यालय",     # conjunct वि + द्या + ल + य
        "हामी",
        "गर्नुहोस्",   # ends with virama (halanta) - valid in Nepali
        "नेपाली",
        "छन्",          # another halanta-ending word
    ]
    for s in test_strings:
        tokens = tokenizer.tokenize_devanagari(s)
        for tok in tokens:
            # If token contains Devanagari, it must be well-formed
            if any(0x0900 <= ord(c) <= 0x097F for c in tok):
                assert is_well_formed_akshara(tok), f"Malformed akshara: {repr(tok)} in {repr(s)}"


def test_no_matra_split():
    """Specific case: म (ma) + ी (i) must be a single token मी, not two."""
    s = "मी"
    tokens = AksharaTokenizer.tokenize_devanagari(s)
    assert tokens == ["मी"], f"Matra split incorrectly: {tokens}"


def test_halanta_tokenization():
    """Words ending in halanta (virama) should be tokenized correctly."""
    s = "गर्नुहोस्"
    tokens = AksharaTokenizer.tokenize_devanagari(s)
    # Should not split the final conjunct
    reconstructed = "".join(tokens)
    assert reconstructed == s, f"Halanta tokenization failed: {tokens}"


# -------------------------------------------------------------------
# 3. Script purity – no token with both DEV and LAT
# -------------------------------------------------------------------
@pytest.fixture
def trained_tokenizer():
    """Minimal training on mixed script data to produce a vocabulary."""
    corpus = """
    मेरो नाम हो। I like to eat rice. स्कुल जान्छु।
    हामी नेपाली हौं। Hello world. विद्यालय जाने बेला भयो।
    """
    tokenizer = NepBPETokenizer(
        frozen_strict={"हरू"},          # minimal strict seed
        frozen_ambiguous=set(),
        punctuation_set={"।", "॥", ",", " "}
    )
    tokenizer.train(corpus, vocab_size=200, latin_budget=20)
    return tokenizer


def test_script_purity(trained_tokenizer):
    """No vocabulary token contains both Devanagari and Latin characters."""
    for token in trained_tokenizer.id2token:
        has_dev = any(0x0900 <= ord(c) <= 0x097F for c in token)
        has_lat = any(c.isascii() and c.isalpha() for c in token)
        if has_dev and has_lat:
            # Byte-fallback tokens like <0x...> are allowed (they are ASCII but not Latin letters)
            if token.startswith("<0x") and token.endswith(">"):
                continue
            pytest.fail(f"Cross-script token found: {repr(token)}")


# -------------------------------------------------------------------
# 4. Frozen morpheme integrity
# -------------------------------------------------------------------
def test_frozen_strict_never_sub_token():
    """V_strict tokens never appear as a proper substring inside a larger token."""
    strict = {"हरू", "हरु"}  # Example: plural morpheme
    tokenizer = NepBPETokenizer(
        frozen_strict=strict,
        punctuation_set={"।", " ", "॥"}
    )
    corpus = "मेरो हरू छन्। तिम्रो हरू पनि छन्।"
    tokenizer.train(corpus, vocab_size=50, latin_budget=0)

    vocab = set(tokenizer.id2token)
    for frozen in strict:
        for token in vocab:
            if token != frozen and frozen in token:
                pytest.fail(
                    f"Frozen morpheme {repr(frozen)} appears inside {repr(token)}"
                )


# -------------------------------------------------------------------
# 5. Byte roundtrip
# -------------------------------------------------------------------
def test_byte_roundtrip_exact():
    """decode_bytes(encode_bytes(original_bytes)) == original_bytes for arbitrary bytes."""
    tokenizer = NepBPETokenizer()
    # Test various byte sequences
    test_cases = [
        b"hello",
        "नमस्ते".encode("utf-8"),
        b"\x00\xff\x80",                # invalid UTF-8 bytes
        b"mix \xe0\xa4\x85",            # partial UTF-8 plus valid
        bytes(range(256)),              # all byte values
    ]
    for orig in test_cases:
        ids = tokenizer.encode_bytes(orig)
        recovered = tokenizer.decode_bytes(ids)
        assert recovered == orig, f"Byte roundtrip failed for {orig!r}"


def test_text_roundtrip_up_to_normalisation():
    """For valid UTF-8 text, decode(encode(text)) == N(text)."""
    tokenizer = NepBPETokenizer()
    texts = [
        "सँग",
        "संग",          # will be normalised
        "hello world",
        "नेपाली",
    ]
    for text in texts:
        ids = tokenizer.encode(text)
        decoded_str = tokenizer.decode(ids)
        expected = normalize(text)
        assert decoded_str == expected, f"Text roundtrip failed for {text!r}: got {decoded_str!r} expected {expected!r}"


def test_space_handling():
    """Spaces should be preserved in encoding/decoding."""
    tokenizer = NepBPETokenizer()
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Space handling failed: {text!r} -> {decoded!r}"


def test_mixed_script_with_spaces():
    """Mixed script text with spaces should roundtrip correctly."""
    tokenizer = NepBPETokenizer(
        punctuation_set={"।", " "}
    )
    corpus = "मेरो नाम हो। Hello world."
    tokenizer.train(corpus, vocab_size=100)
    
    text = "मेरो नाम"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == normalize(text), f"Mixed script roundtrip failed: {text!r} -> {decoded!r}"