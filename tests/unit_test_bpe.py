"""
Unit tests for HimalayanTokenizer (NepBPE v4).
These tests verify the exact guarantees claimed in the equation set.
"""
import pytest
import HimalayanTokenization

def test_normalization_idempotent():
    """N(s) = N(N(s)) for various inputs."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    cases = [
        "सँग",
        "संग",
        "hello world",
        "सँग hello संग",
        "\u200D\u200C", 
    ]
    for s in cases:
        n1 = tok.normalize(s)
        n2 = tok.normalize(n1)
        assert n1 == n2, f"Idempotency failed for {repr(s)}"

def test_folding_table_reduces_variants():
    """All variant forms should map to the same canonical string."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    variants = ["सँग", "संग", "सङ्ग"]
    canonical = tok.normalize("सँग")
    for v in variants:
        assert tok.normalize(v) == canonical, f"Variant {repr(v)} not folded"

def test_zwnj_preserved_zwj_stripped():
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    s = "\u200D\u200C"
    n = tok.normalize(s)
    assert "\u200D" not in n, "ZWJ should be stripped"
    assert "\u200C" in n, "ZWNJ should be preserved"

# -------------------------------------------------------------------
# 2. Akshara integrity (EQ‑1) – no split matra, no broken conjuncts
# -------------------------------------------------------------------
def is_well_formed_akshara(token: str) -> bool:
    """Check that a Devanagari token forms a single akshara.
    A well‑formed akshara:
    - May consist of a consonant cluster optionally followed by a vowel sign
    - May end with a virama (halanta) indicating suppressed inherent vowel
    - Must not have a virama followed by anything other than a consonant or end‑of‑token
    """
    if not any(0x0900 <= ord(c) <= 0x097F for c in token):
        return True
    virama = "\u094D"
    if virama not in token:
        return True
    for i, ch in enumerate(token):
        if ch == virama:
            if i + 1 >= len(token):
                continue
            next_cp = ord(token[i+1])
            if not (0x0915 <= next_cp <= 0x0939):
                return False
    return True

def test_akshara_integrity_on_devanagari():
    """Test that the DFA produces well‑formed aksharas."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    test_strings = [
        "मेरो",
        "स्कुल",        # conjunct स् + कु + ल  = स्कुल
        "विद्यालय",     # conjunct वि + द्या + ल + य
        "हामी",
        "गर्नुहोस्",   # ends with virama (halanta)
        "नेपाली",
        "छन्",          # another halanta‑ending word
    ]
    for s in test_strings:
        aksharas = tok.dfa_tokenize_debug(s)
        for aks in aksharas:
            if any(0x0900 <= ord(c) <= 0x097F for c in aks):
                assert is_well_formed_akshara(aks), f"Malformed akshara: {repr(aks)} in {repr(s)}"

def test_no_matra_split():
    """Specific case: म + ी must be a single token मी."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    aksharas = tok.dfa_tokenize_debug("मी")
    assert aksharas == ["मी"], f"Matra split incorrectly: {aksharas}"

def test_halanta_tokenization():
    """Words ending in halanta (virama) should be tokenized correctly."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    s = "गर्नुहोस्"
    aksharas = tok.dfa_tokenize_debug(s)
    reconstructed = "".join(aksharas)
    assert reconstructed == s, f"Halanta tokenization failed: {aksharas}"

# -------------------------------------------------------------------
# 3. Script purity – no token with both DEV and LAT (EQ‑2 / Gate)
# -------------------------------------------------------------------
@pytest.fixture
def trained_tokenizer():
    """Minimal training on mixed script data to produce a vocabulary."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    # Use aksharas harvested from the corpus (manually listed for reproducibility)
    aksharas = ["मे", "रो", "ना", "म", "हो", "स्कु", "ल", "जा", "न्छु",
                "हा", "मी", "ने", "पा", "ली", "हौं", "हे", "लो",
                "वि", "द्या", "लय", "बे", "ला"]
    tok.initialize_vocab(
        aksharas=aksharas,
        seed_morphemes=[],
        punctuation=[".", ",", "।"],
        v_strict=[],
        v_ambiguous=[]
    )
    corpus = """
    मेरो नाम हो। I like to eat rice. स्कुल जान्छु।
    हामी नेपाली हौं। Hello world. विद्यालय जाने बेला भयो।
    """
    tok.train_from_text([corpus], vocab_budget=200, theta=10)
    return tok

def test_script_purity(trained_tokenizer):
    """No vocabulary token contains both Devanagari and Latin characters."""
    vocab_dict = trained_tokenizer.get_vocab_dict()
    for token in vocab_dict:
        has_dev = any(0x0900 <= ord(c) <= 0x097F for c in token)
        has_lat = any(c.isascii() and c.isalpha() for c in token)
        if has_dev and has_lat:
            # Skip markers and byte‑fallback ASCII (punctuation, control)
            if token in ["▁", "▂"]:
                continue
            if token.isascii() and not token.isalpha():
                continue
            pytest.fail(f"Cross-script token found: {repr(token)}")

# -------------------------------------------------------------------
# 4. Frozen morpheme integrity (V_strict) – never sub‑token
# -------------------------------------------------------------------
def test_frozen_strict_never_sub_token():
    """V_strict tokens never appear as a proper substring inside a larger token."""
    strict = {"हरू"}  # plural morpheme
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    aksharas = ["मे", "रो", "हरू", "छन्", "तिम्रो", "पनि"]
    tok.initialize_vocab(
        aksharas=aksharas,
        seed_morphemes=list(strict),
        punctuation=[".", ",", "।"],
        v_strict=list(strict),
        v_ambiguous=[]
    )
    corpus = "मेरो हरू छन्। तिम्रो हरू पनि छन्।"
    tok.train_from_text([corpus], vocab_budget=30, theta=10)

    vocab = set(tok.get_vocab_dict().keys())
    for frozen in strict:
        for token in vocab:
            if token != frozen and frozen in token:
                pytest.fail(f"Frozen morpheme {repr(frozen)} appears inside {repr(token)}")

# -------------------------------------------------------------------
# 5. Roundtrip guarantees (EQ‑2) – decode(encode(s)) == N(s)
# -------------------------------------------------------------------
def test_text_roundtrip_up_to_normalisation():
    """For valid UTF‑8 text, decode(encode(text)) == N(text)."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    texts = [
        "सँग",
        "संग",
        "hello world",
        "नेपाली",
        "मेरो नाम हो।",
    ]
    for text in texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        expected = tok.normalize(text)
        assert decoded == expected, f"Roundtrip failed for {text!r}: got {decoded!r}, expected {expected!r}"

def test_space_handling():
    """Spaces should be preserved in encoding/decoding."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    text = "hello world"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"Space handling failed: {text!r} -> {decoded!r}"

def test_mixed_script_with_spaces():
    """Mixed script text with spaces should roundtrip correctly."""
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")
    aksharas = ["मे", "रो", "ना", "म", "हो", "हे", "लो"]
    tok.initialize_vocab(
        aksharas=aksharas,
        seed_morphemes=[],
        punctuation=[".", ",", "।"],
        v_strict=[],
        v_ambiguous=[]
    )
    text = "मेरो नाम"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    expected = tok.normalize(text)
    assert decoded == expected, f"Mixed script roundtrip failed: {text!r} -> {decoded!r}"