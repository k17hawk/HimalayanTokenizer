// ============================================================================
// HimalayanTokenization — NepBPE v4
// ----------------------------------------------------------------------------
// This file implements the equations that v2/v3 specified but the previous
// Rust never actually executed. Search for `[EQ-n]` markers to find each one.
//
//   [EQ-1] Phase 2 DFA is WIRED INTO ENCODE. DEV runs are segmented into
//          aksharas and greedy matching happens at AKSHARA granularity, so the
//          "no token splits a matra / breaks a conjunct" guarantee is true by
//          construction instead of aspirational. A built-in Devanagari DFA
//          ships so the machinery is never inert.
//   [EQ-2] Boundary marking M / M^-1 is now a stated, invertible transform, and
//          LITERAL ▁ / ▂ in the input are escaped to MAL so
//          Decode(Encode(s)) = N(s) holds for ALL inputs, not just
//          marker-free ones. Unmarking is per-token by TYPE, not a global
//          string replace.
//   [EQ-3] E1: P(root), the restricted posterior, MorphScore, and W(a) — the
//          full priority key K = (ScriptRank, MorphScore·W·Freq). Ranking only;
//          Legal is untouched, so the existential soundness property survives.
//   [EQ-4] Phase 5: U : TokenID -> RootID ∪ {⊥}, built from the lexicon L
//          (suppletion) and RootSet (allomorphy), exported alongside script(t)
//          so ScriptEmb / ParadigmEmb have real inputs.
//   [EQ-5] E3: folding modes as a BUILD-TIME fork (N_LM / N_OCR), with the mode
//          stamped into the vocab file and checked on load.
//   [EQ-6] E2: induction gate c(u) >= c_lo entering V_ambiguous ONLY, with an
//          explicit human-sign-off promotion path to V_strict.
//   [EQ-7] same_word(i,i+1) enforced on the in-memory training path too.
//   [EQ-8] N idempotence made unconditional (final NFC) + a rule validator.
//   [EQ-9] E4: intrinsic metrics (fertility, byte-fallback rate, vocab
//          efficiency) computed in-engine so eval can't drift from encode.
// ============================================================================

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;

type TokenId = usize;
type RootId = usize;
type ByteVal = u8;
type Frequency = u64;

/// Fixed-point scale for the second component of K. MorphScore ∈ [0,1] and
/// W(a) ∈ R+ are reals; the heap key must be an integer for a total, exactly
/// reproducible order. 1024 keeps ~3 decimal digits of MorphScore resolution
/// while leaving headroom: freq 1e12 · 1024 ≈ 1e15 << u64::MAX.
const PRIORITY_SCALE: f64 = 1024.0;

// ============================================================================
// Boundary markers  [EQ-2]
// ============================================================================

const DEV_MARKER: char = '\u{2581}'; // ▁ precedes Devanagari / other
const LAT_MARKER: char = '\u{2582}'; // ▂ precedes ASCII alphanumeric

#[inline]
fn is_marker(ch: char) -> bool {
    ch == DEV_MARKER || ch == LAT_MARKER
}

/// μ(c): which marker to emit before a boundary, chosen by the FOLLOWING char.
#[inline]
fn marker_for(next: Option<char>) -> char {
    match next {
        Some(c) if c.is_ascii_alphanumeric() => LAT_MARKER,
        _ => DEV_MARKER,
    }
}

/// A character in the marked stream, tagged with whether it is a marker the
/// tokenizer INSERTED (`synthetic`) or a character that was literally present in
/// N(s). This distinction is the whole [EQ-2] fix: a literal U+2581 in the input
/// used to be indistinguishable from an inserted word-start marker, so decode
/// turned it into a space and the roundtrip silently broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MChar {
    ch: char,
    synthetic: bool,
}

const EXTENDED_PUNCT: &[char] = &[
    '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}',
    '\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}',
    '\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}',
    '\u{2026}',
];

#[inline]
fn is_unicode_punct(ch: char) -> bool {
    EXTENDED_PUNCT.contains(&ch)
}

fn unescape_tsv(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\u{0}'; 256];
    let mut n: u32 = 0;
    for b in 0u8..=255u8 {
        let code: u32 = if b.is_ascii_graphic()
            || (0xA1u8..=0xAC).contains(&b)
            || (0xAEu8..=0xFF).contains(&b)
        {
            b as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        table[b as usize] = char::from_u32(code).expect("valid scalar");
    }
    table
}

// ============================================================================
// script(·) — total assignment, §2.1
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    DEV,
    LAT,
    PUN,
    FMT,
    MAL,
}

impl Script {
    fn rank(&self) -> u8 {
        match self {
            Script::DEV => 2,
            Script::PUN => 1,
            _ => 0,
        }
    }
    fn code(&self) -> u8 {
        match self {
            Script::DEV => 0,
            Script::LAT => 1,
            Script::PUN => 2,
            Script::FMT => 3,
            Script::MAL => 4,
        }
    }
}

// ============================================================================
// Phase 1 — Normalization N, mode-indexed  [EQ-5][EQ-8]
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldMode {
    /// N_LM = NFC -> Fold_O^full -> StripZWJ   (ZWNJ preserved)
    LM,
    /// N_OCR = NFC -> Fold_O^minimal -> KeepZWJ+KeepZWNJ   (strictly less lossy)
    OCR,
}

impl FoldMode {
    pub fn tag(&self) -> &'static str {
        match self {
            FoldMode::LM => "LM",
            FoldMode::OCR => "OCR",
        }
    }
    pub fn from_tag(s: &str) -> Option<FoldMode> {
        match s.trim().to_ascii_uppercase().as_str() {
            "LM" => Some(FoldMode::LM),
            "OCR" => Some(FoldMode::OCR),
            _ => None,
        }
    }
    fn strips_zwj(&self) -> bool {
        matches!(self, FoldMode::LM)
    }
}

pub struct Normalizer {
    mode: FoldMode,
    /// Active rule set for this mode, longest pattern first (leftmost-longest).
    fold_rules: Vec<(Vec<char>, String)>,
}

impl Normalizer {
    /// [EQ-5] The mode is chosen HERE, at build time, and is then frozen into
    /// the vocabulary. There is deliberately no runtime setter: flipping the
    /// mode on a trained model would misalign every token id.
    pub fn new_with_mode(
        mode: FoldMode,
        full_rules: Vec<(String, String)>,
        minimal_rules: Vec<(String, String)>,
    ) -> Self {
        let chosen = match mode {
            FoldMode::LM => full_rules,
            FoldMode::OCR => minimal_rules,
        };
        let mut rules: Vec<(Vec<char>, String)> = chosen
            .into_iter()
            .map(|(p, r)| (p.chars().collect::<Vec<char>>(), r))
            .collect();
        rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Self { mode, fold_rules: rules }
    }

    pub fn new(fold_rules: Vec<(String, String)>) -> Self {
        Self::new_with_mode(FoldMode::LM, fold_rules, Vec::new())
    }

    pub fn mode(&self) -> FoldMode {
        self.mode
    }

    /// [EQ-8] Idempotence is now unconditional rather than a caveat: the folded
    /// result is re-composed with NFC before returning, so a rule whose
    /// replacement is not NFC-stable can no longer break N(N(s)) = N(s).
    /// The remaining obligation is that no replacement re-triggers a pattern —
    /// `validate_rules` checks exactly that and is callable from Python.
    pub fn normalize(&self, s: &str) -> String {
        let nfc: String = s.nfc().collect();
        let chars: Vec<char> = nfc.chars().collect();
        let mut result = String::with_capacity(nfc.len());
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;
            for (pat, rep) in &self.fold_rules {
                let plen = pat.len();
                if plen > 0 && i + plen <= chars.len() && chars[i..i + plen] == pat[..] {
                    result.push_str(rep);
                    i += plen;
                    matched = true;
                    break;
                }
            }
            if !matched {
                let ch = chars[i];
                // ZWJ (U+200D) is noise for the LM and signal for OCR.
                // ZWNJ (U+200C) is ALWAYS preserved: it is τ_ZWNJ, an explicit
                // FMT token, in both modes.
                let drop = ch == '\u{200D}' && self.mode.strips_zwj();
                if !drop {
                    result.push(ch);
                }
                i += 1;
            }
        }
        result.nfc().collect()
    }

    /// [EQ-8] Returns the rules that violate the idempotence side condition
    /// (a replacement that itself contains a pattern). Empty = N is idempotent.
    pub fn validate_rules(&self) -> Vec<String> {
        let mut bad = Vec::new();
        for (pat, rep) in &self.fold_rules {
            let pat_s: String = pat.iter().collect();
            for (other, _) in &self.fold_rules {
                let other_s: String = other.iter().collect();
                if !other_s.is_empty() && rep.contains(&other_s) {
                    bad.push(format!("{} -> {} re-triggers {}", pat_s, rep, other_s));
                }
            }
        }
        bad
    }
}

// ============================================================================
// Phase 2 — Akshara DFA  [EQ-1]
// ============================================================================

const VIRAMA: char = '\u{094D}';
const NUKTA: char = '\u{093C}';

fn is_dev_consonant(c: char) -> bool {
    ('\u{0915}'..='\u{0939}').contains(&c)
        || ('\u{0958}'..='\u{095F}').contains(&c)
        || ('\u{0978}'..='\u{097F}').contains(&c)
}
fn is_dev_indep_vowel(c: char) -> bool {
    ('\u{0904}'..='\u{0914}').contains(&c) || ('\u{0972}'..='\u{0977}').contains(&c)
}
fn is_dev_matra(c: char) -> bool {
    ('\u{093A}'..='\u{093B}').contains(&c)
        || ('\u{093E}'..='\u{094C}').contains(&c)
        || c == '\u{094E}'
        || c == '\u{094F}'
        || ('\u{0955}'..='\u{0957}').contains(&c)
        || c == '\u{0962}'
        || c == '\u{0963}'
}
fn is_dev_nasal_mark(c: char) -> bool {
    ('\u{0900}'..='\u{0903}').contains(&c)
}
fn is_dev_digit(c: char) -> bool {
    ('\u{0966}'..='\u{096F}').contains(&c)
}
fn is_dev_standalone(c: char) -> bool {
    c == '\u{093D}' || c == '\u{0950}' || c == '\u{0970}' || c == '\u{0971}'
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Akshara {
    pub surface: String,
    pub root_set: Vec<RootId>,
}

#[derive(Debug, Clone)]
pub struct AksharaDFA {
    pub transitions: HashMap<(usize, char), usize>,
    pub accepting_states: HashSet<usize>,
    initial_state: usize,
}

impl AksharaDFA {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            accepting_states: HashSet::new(),
            initial_state: 0,
        }
    }

    /// [EQ-1] The DFA D the equations always assumed but the code never had.
    /// Previously `AksharaDFA::new()` shipped EMPTY, so `tokenize` degenerated
    /// to one-char-per-unit and the conjunct/matra guarantee was vacuous.
    ///
    /// States: 0 start · 1 after consonant · 2 after virama (half form) ·
    ///         3 after matra · 4 after nasal/visarga · 6 atomic (digit, om, ...)
    /// Accepting: {1, 2, 3, 4, 6}.
    ///
    /// State 2 is ACCEPTING on purpose: a word-final half consonant (क्) is a
    /// complete akshara. With maximal munch this still binds क्+ष into क्ष,
    /// because the longer accepting run wins.
    pub fn devanagari_default() -> Self {
        let mut d = Self::new();
        let  add = |s: usize, c: char, t: usize, acc: bool, d: &mut Self| {
            d.transitions.insert((s, c), t);
            if acc {
                d.accepting_states.insert(t);
            }
        };

        for c in '\u{0900}'..='\u{097F}' {
            if is_dev_consonant(c) {
                add(0, c, 1, true, &mut d);
                add(2, c, 1, true, &mut d); // conjunct: C virama C
            } else if is_dev_indep_vowel(c) {
                add(0, c, 3, true, &mut d);
            } else if is_dev_digit(c) || is_dev_standalone(c) {
                add(0, c, 6, true, &mut d);
            } else if is_dev_matra(c) {
                add(1, c, 3, true, &mut d);
                add(3, c, 3, true, &mut d); // multi-part / stacked matras
            } else if is_dev_nasal_mark(c) {
                add(1, c, 4, true, &mut d);
                add(3, c, 4, true, &mut d);
            }
        }
        add(1, NUKTA, 1, true, &mut d);
        add(1, VIRAMA, 2, true, &mut d);
        d
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn has_transition(&self, state: usize, ch: char) -> Option<usize> {
        self.transitions.get(&(state, ch)).copied()
    }

    pub fn get_transitions_from(&self, state: usize) -> Vec<(char, usize, bool)> {
        self.transitions
            .iter()
            .filter(|((s, _), _)| *s == state)
            .map(|((_, ch), &next)| (*ch, next, self.accepting_states.contains(&next)))
            .collect()
    }

    /// Maximal munch with last-accepting backtracking. A char with no outgoing
    /// transition from the start state is emitted alone (malformed input); it
    /// still becomes its own unit, so no token can straddle it.
    pub fn tokenize(&self, dev_text: &str) -> Vec<Akshara> {
        let mut aksharas = Vec::new();
        let chars: Vec<char> = dev_text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut state = self.initial_state;
            let mut last_accepting_pos: Option<usize> = None;
            let mut j = i;

            while j < chars.len() {
                let ch = chars[j];
                if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                    state = next_state;
                    if self.accepting_states.contains(&state) {
                        last_accepting_pos = Some(j + 1);
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            if let Some(end) = last_accepting_pos {
                aksharas.push(Akshara {
                    surface: chars[i..end].iter().collect(),
                    root_set: Vec::new(),
                });
                i = end;
            } else {
                aksharas.push(Akshara {
                    surface: chars[i].to_string(),
                    root_set: Vec::new(),
                });
                i += 1;
            }
        }
        aksharas
    }
}

// ============================================================================
// Tokens and vocabulary
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Akshara(Akshara),
    Punctuation(String),
    ZWNJ,
    ByteFallback(ByteVal),
    SeededMorpheme(String),
    MergedToken(Vec<TokenId>),
    Latin(String),
    Loaded(String),
}

impl Token {
    pub fn script(&self) -> Script {
        match self {
            Token::Akshara(_) => Script::DEV,
            Token::Punctuation(_) => Script::PUN,
            Token::ZWNJ => Script::FMT,
            Token::ByteFallback(_) => Script::MAL,
            Token::SeededMorpheme(_) => Script::DEV,
            Token::MergedToken(_) => Script::DEV,
            Token::Latin(_) => Script::LAT,
            Token::Loaded(_) => Script::DEV,
        }
    }
}

#[derive(Default)]
pub struct Vocabulary {
    tokens: Vec<Arc<Token>>,
    surface_to_id: HashMap<String, TokenId>,
    id_to_script: HashMap<TokenId, Script>,
    v_strict: HashSet<TokenId>,
    v_ambiguous: HashSet<TokenId>,
    token_to_root_set: HashMap<TokenId, Vec<RootId>>,
    surfaces: HashMap<TokenId, String>,
    max_surface_len: usize,
    /// [EQ-6] c(u) for induced units, kept so promotion decisions are auditable.
    induction_conf: HashMap<TokenId, f64>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(
        &mut self,
        base_aksharas: Vec<Akshara>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: HashSet<String>,
        v_ambiguous: HashSet<String>,
        byte_encoder: &[char; 256],
    ) {
        for akshara in base_aksharas {
            let surface = akshara.surface.clone();
            let token = Arc::new(Token::Akshara(akshara));
            self.add_token(token, surface, false, false);
        }
        for morph in seed_morphemes {
            let surface = morph.clone();
            let is_strict = v_strict.contains(&morph);
            let is_ambiguous = v_ambiguous.contains(&morph);
            let token = Arc::new(Token::SeededMorpheme(morph));
            self.add_token(token, surface, is_strict, is_ambiguous);
        }

        self.add_token(
            Arc::new(Token::SeededMorpheme(DEV_MARKER.to_string())),
            DEV_MARKER.to_string(),
            false,
            false,
        );
        self.add_token(
            Arc::new(Token::Latin(LAT_MARKER.to_string())),
            LAT_MARKER.to_string(),
            false,
            false,
        );

        for punct in punctuation {
            let surface = punct.clone();
            self.add_token(Arc::new(Token::Punctuation(punct)), surface, false, false);
        }
        for &ch in EXTENDED_PUNCT {
            let surface = ch.to_string();
            self.add_token(
                Arc::new(Token::Punctuation(surface.clone())),
                surface,
                false,
                false,
            );
        }
        for ch in ('a'..='z').chain('A'..='Z').chain('0'..='9') {
            let surface = ch.to_string();
            self.add_token(Arc::new(Token::Latin(surface.clone())), surface, false, false);
        }
        for byte_val in 0u8..=255 {
            let surface = byte_encoder[byte_val as usize].to_string();
            self.add_token(Arc::new(Token::ByteFallback(byte_val)), surface, false, false);
        }
        self.add_token(Arc::new(Token::ZWNJ), "\u{200C}".to_string(), false, false);
    }

    fn add_token(
        &mut self,
        token: Arc<Token>,
        surface: String,
        is_strict: bool,
        is_ambiguous: bool,
    ) -> TokenId {
        if let Some(&existing_id) = self.surface_to_id.get(&surface) {
            if is_strict {
                self.v_strict.insert(existing_id);
            }
            if is_ambiguous {
                self.v_ambiguous.insert(existing_id);
            }
            return existing_id;
        }

        let id = self.tokens.len();
        let script = token.script();
        let clen = surface.chars().count();
        if clen > self.max_surface_len {
            self.max_surface_len = clen;
        }
        self.surface_to_id.insert(surface.clone(), id);
        self.id_to_script.insert(id, script);
        self.tokens.push(token);
        self.surfaces.insert(id, surface);

        if is_strict {
            self.v_strict.insert(id);
        }
        if is_ambiguous {
            self.v_ambiguous.insert(id);
        }
        if let Token::Akshara(a) = &*self.tokens[id] {
            self.token_to_root_set.insert(id, a.root_set.clone());
        }
        id
    }

    /// [EQ-6] Induced unit enters as V_ambiguous ONLY. There is no code path
    /// from induction to V_strict; `promote_to_strict` is the only door and it
    /// demands an explicit human-validation flag.
    pub fn add_induced_ambiguous(&mut self, surface: String, confidence: f64) -> TokenId {
        let id = self.add_token(
            Arc::new(Token::SeededMorpheme(surface.clone())),
            surface,
            false,
            true,
        );
        self.induction_conf.insert(id, confidence);
        id
    }

    pub fn promote_to_strict(&mut self, surface: &str, human_validated: bool) -> bool {
        if !human_validated {
            return false;
        }
        match self.surface_to_id.get(surface).copied() {
            Some(id) => {
                self.v_ambiguous.remove(&id);
                self.v_strict.insert(id);
                true
            }
            None => false,
        }
    }

    pub fn induction_confidence(&self, id: TokenId) -> Option<f64> {
        self.induction_conf.get(&id).copied()
    }

    /// §2 tagging. Root sets are stored SORTED BY DESCENDING PRIOR so that the
    /// `root_set_cap` truncation in MorphScore keeps the highest-mass roots
    /// rather than an arbitrary prefix.  [EQ-3]
    pub fn assign_roots_from_registry(
        &mut self,
        registry: &ParadigmRegistry,
        prior: &HashMap<RootId, f64>,
    ) {
        for id in 0..self.tokens.len() {
            let eligible = matches!(
                &*self.tokens[id],
                Token::Akshara(_)
                    | Token::SeededMorpheme(_)
                    | Token::MergedToken(_)
                    | Token::Loaded(_)
            );
            if !eligible {
                continue;
            }
            if let Some(surface) = self.surfaces.get(&id).cloned() {
                let mut roots = registry.get_root_set(&surface);
                if !roots.is_empty() {
                    roots.sort_by(|a, b| {
                        let pa = prior.get(a).copied().unwrap_or(1.0);
                        let pb = prior.get(b).copied().unwrap_or(1.0);
                        pb.partial_cmp(&pa).unwrap_or(Ordering::Equal).then(a.cmp(b))
                    });
                    self.token_to_root_set.insert(id, roots);
                }
            }
        }
    }

    pub fn get_id_by_surface(&self, surface: &str) -> Option<TokenId> {
        self.surface_to_id.get(surface).copied()
    }

    pub fn max_surface_len(&self) -> usize {
        self.max_surface_len
    }

    pub fn load_from_pairs(
        &mut self,
        mut pairs: Vec<(TokenId, String)>,
        byte_decoder: &HashMap<char, u8>,
    ) {
        self.tokens.clear();
        self.surface_to_id.clear();
        self.id_to_script.clear();
        self.v_strict.clear();
        self.v_ambiguous.clear();
        self.token_to_root_set.clear();
        self.surfaces.clear();
        self.induction_conf.clear();
        self.max_surface_len = 0;

        pairs.sort_by_key(|(id, _)| *id);

        for (expected_id, surface) in pairs {
            let id = self.tokens.len();
            debug_assert_eq!(id, expected_id, "vocab ids must be contiguous from 0");

            let mut token = Arc::new(Token::Loaded(surface.clone()));
            let first = surface.chars().next();
            let mut chs = surface.chars();
            if let (Some(ch), None) = (chs.next(), chs.next()) {
                if ch == DEV_MARKER {
                    token = Arc::new(Token::SeededMorpheme(surface.clone()));
                } else if ch == LAT_MARKER {
                    token = Arc::new(Token::Latin(surface.clone()));
                } else if ch == '\u{200C}' {
                    token = Arc::new(Token::ZWNJ);
                } else if ch.is_ascii_alphanumeric() {
                    token = Arc::new(Token::Latin(surface.clone()));
                } else if let Some(&b) = byte_decoder.get(&ch) {
                    token = Arc::new(Token::ByteFallback(b));
                }
            } else if first == Some(LAT_MARKER)
                && surface.chars().skip(1).all(|c| c.is_ascii_alphanumeric())
            {
                token = Arc::new(Token::Latin(surface.clone()));
            } else if surface.chars().all(|c| c.is_ascii_alphanumeric()) {
                token = Arc::new(Token::Latin(surface.clone()));
            }

            let clen = surface.chars().count();
            if clen > self.max_surface_len {
                self.max_surface_len = clen;
            }
            let script = token.script();
            self.surface_to_id.insert(surface.clone(), id);
            self.id_to_script.insert(id, script);
            self.surfaces.insert(id, surface);
            self.tokens.push(token);
        }

        for (m, is_lat) in [(DEV_MARKER, false), (LAT_MARKER, true)] {
            let s = m.to_string();
            if !self.surface_to_id.contains_key(&s) {
                let id = self.tokens.len();
                let tok: Arc<Token> = if is_lat {
                    Arc::new(Token::Latin(s.clone()))
                } else {
                    Arc::new(Token::SeededMorpheme(s.clone()))
                };
                let script = tok.script();
                self.surface_to_id.insert(s.clone(), id);
                self.id_to_script.insert(id, script);
                self.surfaces.insert(id, s);
                self.tokens.push(tok);
            }
        }
    }

    pub fn get_token(&self, id: TokenId) -> Option<&Arc<Token>> {
        self.tokens.get(id)
    }
    pub fn get_script(&self, id: TokenId) -> Script {
        *self.id_to_script.get(&id).unwrap_or(&Script::MAL)
    }
    pub fn is_strict(&self, id: TokenId) -> bool {
        self.v_strict.contains(&id)
    }
    pub fn is_ambiguous(&self, id: TokenId) -> bool {
        self.v_ambiguous.contains(&id)
    }
    pub fn get_root_set(&self, id: TokenId) -> &[RootId] {
        self.token_to_root_set
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn create_merged(
        &mut self,
        a: TokenId,
        b: TokenId,
        merged_root_set: Vec<RootId>,
    ) -> TokenId {
        let surface_a = self.get_surface(a).unwrap_or_default();
        let surface_b = self.get_surface(b).unwrap_or_default();
        let merged_surface = format!("{}{}", surface_a, surface_b);

        let existed = self.surface_to_id.contains_key(&merged_surface);
        let merged = Arc::new(Token::MergedToken(vec![a, b]));
        let id = self.add_token(merged, merged_surface, false, false);

        if !existed {
            if let Some(&script) = self.id_to_script.get(&a) {
                self.id_to_script.insert(id, script);
            }
            self.token_to_root_set.insert(id, merged_root_set);
        }
        id
    }

    pub fn get_surface(&self, id: TokenId) -> Option<String> {
        self.surfaces.get(&id).cloned()
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn get_all_surfaces(&self) -> Vec<(TokenId, String)> {
        self.surfaces.iter().map(|(&id, s)| (id, s.clone())).collect()
    }
    pub fn contains_surface(&self, surface: &str) -> bool {
        self.surface_to_id.contains_key(surface)
    }

    /// [EQ-4] script(t) for every id, in id order — the ScriptEmb input.
    pub fn script_vector(&self) -> Vec<u8> {
        (0..self.tokens.len())
            .map(|id| self.get_script(id).code())
            .collect()
    }

    pub fn count_script(&self, s: Script) -> usize {
        (0..self.tokens.len()).filter(|&id| self.get_script(id) == s).count()
    }
}

// ============================================================================
// Paradigms
// ============================================================================

pub struct Paradigm {
    pub root_id: RootId,
    pub allowed_transitions: HashMap<String, HashSet<String>>,
}

impl Paradigm {
    pub fn new(root_id: RootId) -> Self {
        Self {
            root_id,
            allowed_transitions: HashMap::new(),
        }
    }
    pub fn allows(&self, prefix: &str, continuation: &str) -> bool {
        self.allowed_transitions
            .get(prefix)
            .map(|allowed| allowed.contains(continuation))
            .unwrap_or(false)
    }
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.allowed_transitions.contains_key(prefix)
    }
}

#[derive(Default)]
pub struct ParadigmRegistry {
    paradigms: Vec<Paradigm>,
    root_to_paradigm: HashMap<RootId, usize>,
}

impl ParadigmRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_paradigm(&mut self, paradigm: Paradigm) {
        let idx = self.paradigms.len();
        self.root_to_paradigm.insert(paradigm.root_id, idx);
        self.paradigms.push(paradigm);
    }
    pub fn get_root_set(&self, prefix: &str) -> Vec<RootId> {
        self.paradigms
            .iter()
            .filter(|p| p.has_prefix(prefix))
            .map(|p| p.root_id)
            .collect()
    }
    pub fn check_allowed(&self, root_id: RootId, prefix: &str, continuation: &str) -> bool {
        if let Some(&idx) = self.root_to_paradigm.get(&root_id) {
            self.paradigms[idx].allows(prefix, continuation)
        } else {
            false
        }
    }
    pub fn roots(&self) -> Vec<RootId> {
        self.paradigms.iter().map(|p| p.root_id).collect()
    }
}

// ============================================================================
// Phase 3 — corpus + constrained BPE
// ============================================================================

fn dec_pair(
    freqs: &mut HashMap<(TokenId, TokenId), Frequency>,
    key: (TokenId, TokenId),
    by: Frequency,
) {
    if let Some(f) = freqs.get_mut(&key) {
        *f = f.saturating_sub(by);
        if *f == 0 {
            freqs.remove(&key);
        }
    }
}

fn inc_pair(
    freqs: &mut HashMap<(TokenId, TokenId), Frequency>,
    key: (TokenId, TokenId),
    by: Frequency,
) {
    *freqs.entry(key).or_insert(0) += by;
}

struct Word {
    tokens: Vec<TokenId>,
    count: Frequency,
}

pub struct Corpus {
    words: Vec<Word>,
    pair_freqs: HashMap<(TokenId, TokenId), Frequency>,
    #[allow(dead_code)]
    vocab_budget: usize,
}

impl Corpus {
    /// [EQ-7] Every entry here is one WORD TYPE, so `windows(2)` can never
    /// straddle a word boundary — this is what makes the `same_word(i,i+1)`
    /// side condition on Freq true rather than assumed.
    pub fn from_word_counts(
        counts: HashMap<Vec<TokenId>, Frequency>,
        vocab_budget: usize,
    ) -> Self {
        let words: Vec<Word> = counts
            .into_iter()
            .map(|(tokens, count)| Word { tokens, count })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    pub fn from_sequences(sequences: Vec<Vec<TokenId>>, vocab_budget: usize) -> Self {
        let words: Vec<Word> = sequences
            .into_iter()
            .map(|tokens| Word { tokens, count: 1 })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    fn recompute_all_frequencies(&mut self) {
        self.pair_freqs.clear();
        for w in &self.words {
            for window in w.tokens.windows(2) {
                *self.pair_freqs.entry((window[0], window[1])).or_insert(0) += w.count;
            }
        }
    }

    pub fn get_freq(&self, a: TokenId, b: TokenId) -> Frequency {
        self.pair_freqs.get(&(a, b)).copied().unwrap_or(0)
    }

    pub fn apply_merge(
        &mut self,
        a: TokenId,
        b: TokenId,
        new_id: TokenId,
    ) -> Vec<(TokenId, TokenId)> {
        let mut touched: HashSet<(TokenId, TokenId)> = HashSet::new();

        for w in self.words.iter_mut() {
            let cnt = w.count;
            let mut i = 0;
            while i + 1 < w.tokens.len() {
                if w.tokens[i] == a && w.tokens[i + 1] == b {
                    if i > 0 {
                        let l = w.tokens[i - 1];
                        dec_pair(&mut self.pair_freqs, (l, a), cnt);
                        touched.insert((l, a));
                    }
                    if i + 2 < w.tokens.len() {
                        let r = w.tokens[i + 2];
                        dec_pair(&mut self.pair_freqs, (b, r), cnt);
                        touched.insert((b, r));
                    }

                    w.tokens[i] = new_id;
                    w.tokens.remove(i + 1);

                    if i > 0 {
                        let l = w.tokens[i - 1];
                        inc_pair(&mut self.pair_freqs, (l, new_id), cnt);
                        touched.insert((l, new_id));
                    }
                    if i + 1 < w.tokens.len() {
                        let r = w.tokens[i + 1];
                        inc_pair(&mut self.pair_freqs, (new_id, r), cnt);
                        touched.insert((new_id, r));
                    }
                } else {
                    i += 1;
                }
            }
        }

        self.pair_freqs.remove(&(a, b));
        touched.into_iter().collect()
    }
}

#[derive(Debug, Clone)]
struct MergeCandidate {
    a: TokenId,
    b: TokenId,
    priority_key: (u8, u64),
    freq_snapshot: Frequency,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority_key == other.priority_key && self.a == other.a && self.b == other.b
    }
}
impl Eq for MergeCandidate {}
impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_key
            .cmp(&other.priority_key)
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

pub struct ConstrainedBPETrainer {
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    pub theta: Frequency,
    pub latin_pass: bool,
    /// [EQ-3] P(root): root-frequency prior from FST analysis (E5). Absent
    /// entries default to 1.0, which makes the posterior uniform over RootSet —
    /// i.e. E1 with no prior degrades to "fraction of consistent roots that
    /// license b", never to a crash.
    pub root_prior: HashMap<RootId, f64>,
    /// [EQ-3] W(a): optional intra-tier weight, default 1.
    pub token_weight: HashMap<TokenId, f64>,
    /// [EQ-3] E1 on/off. OFF reproduces v2's K exactly (MorphScore = W = 1).
    pub use_probabilistic_k: bool,
    /// [EQ-3] Cap on |RootSet| scanned by MorphScore. 0 = uncapped.
    pub root_set_cap: usize,
}

impl ConstrainedBPETrainer {
    pub fn new(vocab: Vocabulary, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            vocab,
            paradigm_registry,
            theta: 100,
            latin_pass: false,
            root_prior: HashMap::new(),
            token_weight: HashMap::new(),
            use_probabilistic_k: false,
            root_set_cap: 0,
        }
    }

    fn script_compat(&self, a: TokenId, b: TokenId) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);
        if self.latin_pass {
            return sa == sb && sa == Script::LAT;
        }
        sa == sb && (sa == Script::DEV || sa == Script::PUN)
    }

    fn gate(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);

        if sa == Script::MAL || sa == Script::FMT || sb == Script::MAL || sb == Script::FMT {
            return false;
        }
        if !self.script_compat(a, b) {
            return false;
        }
        if self.latin_pass {
            return true;
        }
        if self.vocab.is_strict(b) || self.vocab.is_ambiguous(b) {
            return false; // b ∈ V_seed
        }
        if self.vocab.is_strict(a) {
            return false; // V_strict is terminal
        }
        if self.vocab.is_ambiguous(a) && freq < self.theta {
            return false;
        }
        true
    }

    /// Hard, existential, UNCHANGED by E1. This is the one soundness property
    /// in the system and E1 deliberately does not touch it.
    fn morph(&self, a: TokenId, b: TokenId) -> bool {
        if self.latin_pass {
            return true;
        }
        let root_set = self.vocab.get_root_set(a);
        if root_set.is_empty() {
            return true;
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();
        root_set
            .iter()
            .any(|&root_id| self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b))
    }

    fn legal(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        self.gate(a, b, freq) && self.morph(a, b)
    }

    /// [EQ-3] MorphScore(a,b) = posterior MASS of consistent roots licensing b.
    ///   P(root | hist(a)) ∝ P(root) · 1[root ∈ RootSet(a)]
    ///   MorphScore      = Σ_{root ∈ RootSet(a)} P(root|hist(a)) · 1[b allowed]
    /// Returns 1.0 for non-paradigm tokens (neutral) and 1.0 for a degenerate
    /// all-zero prior, so it can never zero out an otherwise admissible merge.
    fn morph_score(&self, a: TokenId, b: TokenId) -> f64 {
        if self.latin_pass || !self.use_probabilistic_k {
            return 1.0;
        }
        let root_set = self.vocab.get_root_set(a);
        if root_set.is_empty() {
            return 1.0;
        }
        let n = if self.root_set_cap == 0 {
            root_set.len()
        } else {
            self.root_set_cap.min(root_set.len())
        };
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();

        let mut total = 0.0f64;
        let mut licensed = 0.0f64;
        for &r in &root_set[..n] {
            let p = self.root_prior.get(&r).copied().unwrap_or(1.0).max(0.0);
            total += p;
            if self.paradigm_registry.check_allowed(r, &surface_a, &surface_b) {
                licensed += p;
            }
        }
        if total <= 0.0 {
            1.0
        } else {
            licensed / total
        }
    }

    fn narrow_root_set(&self, a: TokenId, b: TokenId) -> Vec<RootId> {
        if self.latin_pass {
            return Vec::new();
        }
        let root_set_a = self.vocab.get_root_set(a);
        if root_set_a.is_empty() {
            return Vec::new();
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();
        root_set_a
            .iter()
            .filter(|&&root_id| {
                self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b)
            })
            .copied()
            .collect()
    }

    /// [EQ-3] K(a,b) = ( ScriptRank(script(a)), MorphScore(a,b)·W(a)·Freq(a,b) )
    /// under LEXICOGRAPHIC order. The second component is fixed-point so the
    /// heap order is total and bit-reproducible across runs.
    fn priority_key(&self, a: TokenId, b: TokenId, freq: Frequency) -> (u8, u64) {
        let rank = self.vocab.get_script(a).rank();
        let ms = self.morph_score(a, b);
        let w = self.token_weight.get(&a).copied().unwrap_or(1.0).max(0.0);
        let scaled = ms * w * (freq as f64) * PRIORITY_SCALE;
        let key = if scaled.is_finite() && scaled > 0.0 {
            scaled.min(u64::MAX as f64) as u64
        } else {
            0
        };
        (rank, key)
    }

    pub fn train(&mut self, corpus: &mut Corpus, budget: usize, progress_every: u64) {
        let t0 = std::time::Instant::now();
        let start_vocab = self.vocab.len();
        let tag = if self.latin_pass { "LAT" } else { "DEV" };

        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        self.initialize_heap(corpus, &mut heap);
        if progress_every > 0 {
            eprintln!(
                "[train:{}] heap seeded with {} admissible pairs in {:.1}s",
                tag,
                heap.len(),
                t0.elapsed().as_secs_f64()
            );
        }

        let mut merges: u64 = 0;

        while self.vocab.len() < budget {
            let mut applied = false;

            while let Some(candidate) = heap.pop() {
                let current_freq = corpus.get_freq(candidate.a, candidate.b);
                if current_freq != candidate.freq_snapshot {
                    continue;
                }
                if current_freq == 0 {
                    continue;
                }
                if !self.legal(candidate.a, candidate.b, current_freq) {
                    continue;
                }

                let narrowed = self.narrow_root_set(candidate.a, candidate.b);
                let new_id = self.vocab.create_merged(candidate.a, candidate.b, narrowed);
                let touched = corpus.apply_merge(candidate.a, candidate.b, new_id);

                for (x, y) in touched {
                    let f = corpus.get_freq(x, y);
                    if f > 0 && self.legal(x, y, f) {
                        heap.push(MergeCandidate {
                            a: x,
                            b: y,
                            priority_key: self.priority_key(x, y, f),
                            freq_snapshot: f,
                        });
                    }
                }

                merges += 1;
                applied = true;

                if progress_every > 0 && merges % progress_every == 0 {
                    let secs = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[train:{}] {} merges | vocab {} | heap {} | {:.1}s | {:.0} merges/s",
                        tag,
                        merges,
                        self.vocab.len(),
                        heap.len(),
                        secs,
                        merges as f64 / secs.max(1e-9)
                    );
                }
                break;
            }

            if !applied {
                break;
            }
        }

        if progress_every > 0 {
            eprintln!(
                "[train:{}] done: {} merges ({} -> {} vocab) in {:.1}s",
                tag,
                merges,
                start_vocab,
                self.vocab.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    fn initialize_heap(&self, corpus: &Corpus, heap: &mut BinaryHeap<MergeCandidate>) {
        for &(a, b) in corpus.pair_freqs.keys() {
            let freq = corpus.get_freq(a, b);
            if self.legal(a, b, freq) {
                heap.push(MergeCandidate {
                    a,
                    b,
                    priority_key: self.priority_key(a, b, freq),
                    freq_snapshot: freq,
                });
            }
        }
    }
}

// ============================================================================
// Main tokenizer
// ============================================================================

#[allow(dead_code)]
#[allow(non_camel_case_types)]
pub struct HimalayanTokenization {
    pub normalizer: Normalizer,
    pub akshara_dfa: AksharaDFA,
    pub vocab: Vocabulary,
    pub paradigm_registry: ParadigmRegistry,
    byte_encoder: [char; 256],
    seed_max_len: usize,

    // ---- E1 / E5 state  [EQ-3] ----
    root_prior: HashMap<RootId, f64>,
    token_weight: HashMap<TokenId, f64>,
    use_probabilistic_k: bool,
    root_set_cap: usize,

    // ---- Phase 5 state  [EQ-4] ----
    /// L: surface -> RootID. The ONLY possible mechanism for suppletion
    /// (हुनु -> भयो/छ/थियो, जानु -> गयो share no substring with their root).
    lexicon: HashMap<String, RootId>,
    /// U : TokenID -> RootID ∪ {⊥}
    paradigm_embedding: HashMap<TokenId, Option<RootId>>,
}

impl HimalayanTokenization {
    pub fn new(fold_rules: Vec<(String, String)>, paradigm_registry: ParadigmRegistry) -> Self {
        Self::new_with_mode(FoldMode::LM, fold_rules, Vec::new(), paradigm_registry)
    }

    pub fn new_with_mode(
        mode: FoldMode,
        full_rules: Vec<(String, String)>,
        minimal_rules: Vec<(String, String)>,
        paradigm_registry: ParadigmRegistry,
    ) -> Self {
        Self {
            normalizer: Normalizer::new_with_mode(mode, full_rules, minimal_rules),
            // [EQ-1] The DFA is populated by default; `clear_dfa` + explicit
            // transitions can still override it for a custom orthography.
            akshara_dfa: AksharaDFA::devanagari_default(),
            vocab: Vocabulary::new(),
            paradigm_registry,
            byte_encoder: bytes_to_unicode(),
            seed_max_len: 0,
            root_prior: HashMap::new(),
            token_weight: HashMap::new(),
            use_probabilistic_k: false,
            root_set_cap: 0,
            lexicon: HashMap::new(),
            paradigm_embedding: HashMap::new(),
        }
    }

    pub fn encode(&self, s: &str) -> Vec<TokenId> {
        let normalized = self.normalizer.normalize(s);
        self.encode_normalized(&normalized)
    }

    pub fn load_vocab(&mut self, pairs: Vec<(TokenId, String)>) {
        let mut byte_decoder: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &ch) in self.byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, b as u8);
        }

        let mut seed_max_len = 0usize;
        for (_, surface) in &pairs {
            if surface.contains('.') || surface.contains('\u{0964}') {
                let clen = surface.chars().count();
                if clen >= 2 && clen > seed_max_len {
                    seed_max_len = clen;
                }
            }
        }
        self.seed_max_len = seed_max_len;
        self.vocab.load_from_pairs(pairs, &byte_decoder);
    }

    // ------------------------------------------------------------------
    // [EQ-2] Mark / Unmark
    // ------------------------------------------------------------------

    /// M(s): prepend a dummy word-start marker and replace every ASCII space
    /// with the marker chosen by lookahead. Characters that were LITERALLY ▁/▂
    /// in N(s) are carried through with `synthetic = false` and later routed to
    /// MAL, so M is injective and M^-1 is exact.
    fn mark(&self, normalized: &str) -> Vec<MChar> {
        let src: Vec<char> = normalized.chars().collect();
        let mut marked: Vec<MChar> = Vec::with_capacity(src.len() + 1);
        marked.push(MChar {
            ch: marker_for(src.first().copied()),
            synthetic: true,
        });
        for (idx, &ch) in src.iter().enumerate() {
            if ch == ' ' {
                marked.push(MChar {
                    ch: marker_for(src.get(idx + 1).copied()),
                    synthetic: true,
                });
            } else {
                marked.push(MChar { ch, synthetic: false });
            }
        }
        marked
    }

    fn classify_mchar(&self, m: MChar) -> Script {
        if m.synthetic {
            if m.ch == DEV_MARKER {
                Script::DEV
            } else {
                Script::LAT
            }
        } else if is_marker(m.ch) {
            // [EQ-2] A literal marker character can never be allowed to become
            // a marker token — it would decode to a space. Route it to MAL so it
            // survives as raw bytes.
            Script::MAL
        } else {
            self.classify_char(m.ch)
        }
    }

    /// Atomic abbreviation pre-scan, over the marked slice. The candidate window
    /// stops at the first synthetic marker: an abbreviation cannot span a space.
    fn try_emit_atomic_seed(
        &self,
        marked: &[MChar],
        start: usize,
        tokens: &mut Vec<TokenId>,
    ) -> Option<usize> {
        let n = marked.len();
        if self.seed_max_len == 0 || start >= n {
            return None;
        }

        let window = self.seed_max_len.min(n - start);
        let has_punct = marked[start..start + window]
            .iter()
            .any(|m| !m.synthetic && (m.ch == '.' || m.ch == '\u{0964}'));
        if !has_punct {
            return None;
        }

        let leading_marker = if marked[start].synthetic {
            Some(marked[start].ch)
        } else {
            None
        };
        let offset = if leading_marker.is_some() { 1 } else { 0 };
        if start + offset >= n {
            return None;
        }

        // Longest run of non-synthetic chars available from here.
        let mut avail = 0usize;
        while start + offset + avail < n && !marked[start + offset + avail].synthetic {
            avail += 1;
        }
        let max_match = self.seed_max_len.min(avail);
        if max_match < 2 {
            return None;
        }

        for len in (2..=max_match).rev() {
            let candidate: String = marked[start + offset..start + offset + len]
                .iter()
                .map(|m| m.ch)
                .collect();
            if let Some(seed_id) = self.vocab.get_id_by_surface(&candidate) {
                if let Some(m) = leading_marker {
                    if let Some(marker_id) = self.vocab.get_id_by_surface(&m.to_string()) {
                        tokens.push(marker_id);
                    }
                }
                tokens.push(seed_id);
                return Some(start + offset + len);
            }
        }
        None
    }

    pub fn encode_normalized(&self, normalized: &str) -> Vec<TokenId> {
        let marked = self.mark(normalized);
        let n = marked.len();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < n {
            if let Some(end) = self.try_emit_atomic_seed(&marked, i, &mut tokens) {
                i = end;
                continue;
            }

            let start = i;
            let first_script = self.classify_mchar(marked[i]);
            i += 1;
            // A run never crosses a SYNTHETIC marker (that would fuse two words);
            // literal markers are MAL so they break the run on script alone.
            while i < n && self.classify_mchar(marked[i]) == first_script && !marked[i].synthetic {
                i += 1;
            }
            self.tokenize_run(&marked[start..i], first_script, &mut tokens);
        }

        tokens
    }

    fn classify_char(&self, ch: char) -> Script {
        if ch == '\u{200C}' {
            Script::FMT
        } else if ch.is_ascii_punctuation()
            || ch == '\u{0964}'
            || ch == '\u{0965}'
            || is_unicode_punct(ch)
        {
            Script::PUN
        } else if ch.is_ascii_alphabetic() || ch.is_ascii_digit() {
            Script::LAT
        } else if ('\u{0900}'..='\u{097F}').contains(&ch) {
            Script::DEV
        } else {
            Script::MAL
        }
    }

    /// [EQ-1] The unit alphabet of a run. For DEV this is the DFA's akshara
    /// segmentation (plus a leading marker unit if present); for LAT it is
    /// single characters. Greedy matching operates on WHOLE units, which is what
    /// makes "no token splits a matra or breaks a conjunct" hold at encode time
    /// and not just in Phase 2's intent.
    fn run_units(&self, run: &[MChar], script: Script) -> Vec<String> {
        match script {
            Script::DEV => {
                let mut units: Vec<String> = Vec::new();
                let mut rest = String::new();
                for (k, m) in run.iter().enumerate() {
                    if k == 0 && m.synthetic {
                        units.push(m.ch.to_string());
                    } else {
                        rest.push(m.ch);
                    }
                }
                if !rest.is_empty() {
                    for a in self.akshara_dfa.tokenize(&rest) {
                        units.push(a.surface);
                    }
                }
                units
            }
            _ => run.iter().map(|m| m.ch.to_string()).collect(),
        }
    }

    fn tokenize_run(&self, run: &[MChar], script: Script, tokens: &mut Vec<TokenId>) {
        match script {
            Script::DEV | Script::LAT => {
                let units = self.run_units(run, script);
                self.tokenize_greedy_units(&units, tokens);
            }
            Script::PUN => {
                for m in run {
                    let s = m.ch.to_string();
                    if let Some(id) = self.vocab.get_id_by_surface(&s) {
                        tokens.push(id);
                    } else {
                        self.emit_byte_fallback(s.as_bytes(), tokens);
                    }
                }
            }
            Script::FMT => {
                for _ in run {
                    if let Some(id) = self.vocab.get_id_by_surface("\u{200C}") {
                        tokens.push(id);
                    }
                }
            }
            Script::MAL => {
                let s: String = run.iter().map(|m| m.ch).collect();
                self.emit_byte_fallback(s.as_bytes(), tokens);
            }
        }
    }

    /// Greedy longest-match over concatenations of WHOLE units. Forward scan
    /// with best-match tracking: O(cap) surface lookups per position, and the
    /// emitted boundaries are always a subset of the unit boundaries.
    fn tokenize_greedy_units(&self, units: &[String], tokens: &mut Vec<TokenId>) {
        let n = units.len();
        let cap = self.vocab.max_surface_len().max(1);
        let mut i = 0;

        while i < n {
            let mut cand = String::new();
            let mut clen = 0usize;
            let mut best: Option<(TokenId, usize)> = None;
            let mut j = i;

            while j < n {
                let ulen = units[j].chars().count();
                if clen + ulen > cap {
                    break;
                }
                cand.push_str(&units[j]);
                clen += ulen;
                j += 1;
                if let Some(id) = self.vocab.get_id_by_surface(&cand) {
                    best = Some((id, j));
                }
            }

            match best {
                Some((id, end)) => {
                    tokens.push(id);
                    i = end;
                }
                None => {
                    // Unknown unit -> spell the WHOLE unit out in bytes. Note
                    // this still never produces a DEV token straddling part of
                    // an akshara: the fallback tokens are MAL and cover the unit
                    // exactly.
                    self.emit_byte_fallback(units[i].as_bytes(), tokens);
                    i += 1;
                }
            }
        }
    }

    fn emit_byte_fallback(&self, bytes: &[u8], tokens: &mut Vec<TokenId>) {
        for &byte in bytes {
            let surface = self.byte_encoder[byte as usize].to_string();
            if let Some(id) = self.vocab.get_id_by_surface(&surface) {
                tokens.push(id);
            } else {
                debug_assert!(false, "byte-fallback alphabet incomplete for 0x{:02X}", byte);
            }
        }
    }

    /// [EQ-2] M^-1. Unmarking is applied PER TOKEN and only to non-byte token
    /// surfaces. Byte-fallback content is flushed verbatim, so a literal ▁ in
    /// the source comes back as ▁ rather than as a space. Then the single dummy
    /// prefix space is dropped.
    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in token_ids {
            match self.vocab.get_token(id).map(|arc| arc.as_ref()) {
                Some(Token::ByteFallback(byte)) => {
                    byte_buf.push(*byte);
                }
                Some(_) => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                    if let Some(surface) = self.vocab.get_surface(id) {
                        for ch in surface.chars() {
                            if is_marker(ch) {
                                result.push(' ');
                            } else {
                                result.push(ch);
                            }
                        }
                    }
                }
                None => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                }
            }
        }
        if !byte_buf.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        match result.strip_prefix(' ') {
            Some(rest) => rest.to_string(),
            None => result,
        }
    }

    pub fn verify_roundtrip(&self, s: &str) -> bool {
        let encoded = self.encode(s);
        self.decode(&encoded) == self.normalizer.normalize(s)
    }

    /// [EQ-4] Build U : TokenID -> RootID ∪ {⊥}.
    ///   1. L (lexicon) wins — it is the only thing that can express suppletion.
    ///   2. |RootSet(t)| = 1 -> that root (narrowing has disambiguated).
    ///   3. |RootSet(t)| > 1 -> argmax P(root) if `seed_ambiguous_by_prior`,
    ///      else ⊥. Either way this is a SEED; U is learned downstream.
    ///   4. otherwise ⊥.
    pub fn build_paradigm_embedding(&mut self, seed_ambiguous_by_prior: bool) -> usize {
        self.paradigm_embedding.clear();
        let mut assigned = 0usize;
        for id in 0..self.vocab.len() {
            let surface = self.vocab.get_surface(id).unwrap_or_default();
            let u = if let Some(&r) = self.lexicon.get(&surface) {
                Some(r)
            } else {
                let rs = self.vocab.get_root_set(id);
                if rs.len() == 1 {
                    Some(rs[0])
                } else if rs.len() > 1 && seed_ambiguous_by_prior {
                    rs.iter()
                        .copied()
                        .max_by(|a, b| {
                            let pa = self.root_prior.get(a).copied().unwrap_or(1.0);
                            let pb = self.root_prior.get(b).copied().unwrap_or(1.0);
                            pa.partial_cmp(&pb).unwrap_or(Ordering::Equal).then(b.cmp(a))
                        })
                } else {
                    None
                }
            };
            if u.is_some() {
                assigned += 1;
            }
            self.paradigm_embedding.insert(id, u);
        }
        assigned
    }

    pub fn u_vector(&self) -> Vec<i64> {
        (0..self.vocab.len())
            .map(|id| match self.paradigm_embedding.get(&id) {
                Some(Some(r)) => *r as i64,
                _ => -1,
            })
            .collect()
    }
}

// ============================================================================
// Python bindings
// ============================================================================

#[pyclass]
#[allow(non_camel_case_types)]
pub struct PyHimalayanTokenization {
    inner: HimalayanTokenization,
}

impl PyHimalayanTokenization {
    fn make_trainer(&mut self) -> ConstrainedBPETrainer {
        let mut t = ConstrainedBPETrainer::new(
            std::mem::take(&mut self.inner.vocab),
            std::mem::take(&mut self.inner.paradigm_registry),
        );
        t.root_prior = self.inner.root_prior.clone();
        t.token_weight = self.inner.token_weight.clone();
        t.use_probabilistic_k = self.inner.use_probabilistic_k;
        t.root_set_cap = self.inner.root_set_cap;
        t
    }

    fn reclaim(&mut self, trainer: ConstrainedBPETrainer) {
        self.inner.vocab = trainer.vocab;
        self.inner.paradigm_registry = trainer.paradigm_registry;
    }

    fn tag_roots(&mut self) {
        let prior = self.inner.root_prior.clone();
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry, &prior);
    }
}

#[pymethods]
impl PyHimalayanTokenization {
    /// [EQ-5] `mode` is a BUILD-TIME fork. "LM" applies the full folding table
    /// and strips ZWJ; "OCR" applies only the minimal table and preserves both
    /// ZWJ and ZWNJ. The resulting vocabulary is mode-specific and is stamped
    /// with the mode by `save_vocab_tsv`.
    #[new]
    #[pyo3(signature = (folding_rules=None, minimal_folding_rules=None, mode="LM"))]
    fn new(
        folding_rules: Option<Vec<(String, String)>>,
        minimal_folding_rules: Option<Vec<(String, String)>>,
        mode: &str,
    ) -> PyResult<Self> {
        let m = FoldMode::from_tag(mode).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("mode must be 'LM' or 'OCR'")
        })?;
        let inner = HimalayanTokenization::new_with_mode(
            m,
            folding_rules.unwrap_or_default(),
            minimal_folding_rules.unwrap_or_default(),
            ParadigmRegistry::new(),
        );
        Ok(Self { inner })
    }

    fn fold_mode(&self) -> String {
        self.inner.normalizer.mode().tag().to_string()
    }

    fn normalize(&self, s: &str) -> String {
        self.inner.normalizer.normalize(s)
    }

    /// [EQ-8] Empty list => N is idempotent for the configured rule set.
    fn validate_folding_rules(&self) -> Vec<String> {
        self.inner.normalizer.validate_rules()
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        Ok(self.inner.encode(text))
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn verify_roundtrip(&self, s: &str) -> bool {
        self.inner.verify_roundtrip(s)
    }

    // ---------------- vocabulary ----------------

    fn initialize_vocab(
        &mut self,
        aksharas: Vec<String>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: Vec<String>,
        v_ambiguous: Vec<String>,
    ) -> PyResult<usize> {
        let base_aksharas: Vec<Akshara> = aksharas
            .into_iter()
            .map(|s| Akshara {
                surface: s,
                root_set: Vec::new(),
            })
            .collect();

        let mut seed_max_len = 0usize;
        for surf in &seed_morphemes {
            if surf.contains('.') || surf.contains('\u{0964}') {
                seed_max_len = seed_max_len.max(surf.chars().count());
            }
        }
        self.inner.seed_max_len = seed_max_len;

        self.inner.vocab.initialize(
            base_aksharas,
            seed_morphemes,
            punctuation,
            v_strict.into_iter().collect(),
            v_ambiguous.into_iter().collect(),
            &self.inner.byte_encoder,
        );
        Ok(self.inner.vocab.len())
    }

    /// [EQ-1] Stream a corpus and return the akshara inventory the DFA actually
    /// produces, most frequent first. Feed this to `initialize_vocab(aksharas=…)`
    /// so BaseAksharas covers the corpus and unseen-akshara byte fallback stays
    /// near zero.
    #[pyo3(signature = (path, min_freq=1))]
    fn harvest_aksharas(
        &self,
        py: Python<'_>,
        path: String,
        min_freq: u64,
    ) -> PyResult<Vec<String>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        let out = py.allow_threads(|| {
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<String, u64> = HashMap::new();
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let norm = self.inner.normalizer.normalize(&line);
                let mut run = String::new();
                for ch in norm.chars().chain(std::iter::once('\n')) {
                    if ('\u{0900}'..='\u{097F}').contains(&ch)
                        && ch != '\u{0964}'
                        && ch != '\u{0965}'
                    {
                        run.push(ch);
                    } else if !run.is_empty() {
                        for a in self.inner.akshara_dfa.tokenize(&run) {
                            *counts.entry(a.surface).or_insert(0) += 1;
                        }
                        run.clear();
                    }
                }
            }
            let mut v: Vec<(String, u64)> =
                counts.into_iter().filter(|(_, c)| *c >= min_freq).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v.into_iter().map(|(s, _)| s).collect::<Vec<String>>()
        });
        Ok(out)
    }

    fn clear_dfa(&mut self) {
        self.inner.akshara_dfa = AksharaDFA::new();
    }

    fn add_dfa_transition(&mut self, state: usize, ch: char, next_state: usize, accepting: bool) {
        self.inner.akshara_dfa.transitions.insert((state, ch), next_state);
        if accepting {
            self.inner.akshara_dfa.accepting_states.insert(next_state);
        }
    }

    fn dfa_has_transition(&self, state: usize, ch: char) -> bool {
        self.inner.akshara_dfa.has_transition(state, ch).is_some()
    }

    fn dfa_get_transitions(&self, state: usize) -> Vec<(String, usize, bool)> {
        self.inner
            .akshara_dfa
            .get_transitions_from(state)
            .into_iter()
            .map(|(ch, next, acc)| (ch.to_string(), next, acc))
            .collect()
    }

    fn dfa_tokenize_debug(&self, text: &str) -> Vec<String> {
        self.inner
            .akshara_dfa
            .tokenize(text)
            .into_iter()
            .map(|a| a.surface)
            .collect()
    }

    fn vocab_contains(&self, surface: &str) -> bool {
        self.inner.vocab.contains_surface(surface)
    }
    fn vocab_get_id(&self, surface: &str) -> Option<usize> {
        self.inner.vocab.get_id_by_surface(surface)
    }
    fn vocab_get_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab.len()
    }

    // ---------------- paradigms / E1 / E5 ----------------

    fn add_paradigm(&mut self, root_id: usize, transitions: Bound<'_, PyDict>) -> PyResult<()> {
        let mut paradigm = Paradigm::new(root_id);
        for (prefix, continuations) in transitions.iter() {
            let prefix_str: String = prefix.extract()?;
            let cont_list: Vec<String> = continuations.extract()?;
            paradigm
                .allowed_transitions
                .insert(prefix_str, cont_list.into_iter().collect());
        }
        self.inner.paradigm_registry.add_paradigm(paradigm);
        Ok(())
    }

    /// [EQ-3] P(root): root-frequency prior, normally derived from FST analysis
    /// over the corpus (E5). Values need not sum to 1 — the posterior
    /// renormalizes over RootSet(a).
    fn set_root_prior(&mut self, prior: HashMap<usize, f64>) {
        self.inner.root_prior = prior;
    }

    /// [EQ-3] W(a), keyed by surface. Default 1 for anything unset.
    fn set_token_weight(&mut self, surface: &str, weight: f64) -> bool {
        match self.inner.vocab.get_id_by_surface(surface) {
            Some(id) => {
                self.inner.token_weight.insert(id, weight);
                true
            }
            None => false,
        }
    }

    /// [EQ-3] Toggle E1. OFF => K reduces exactly to v2's (ScriptRank, Freq).
    #[pyo3(signature = (enabled, root_set_cap=0))]
    fn enable_probabilistic_k(&mut self, enabled: bool, root_set_cap: usize) {
        self.inner.use_probabilistic_k = enabled;
        self.inner.root_set_cap = root_set_cap;
    }

    fn assign_initial_roots(&mut self) {
        self.tag_roots();
    }

    fn get_root_set(&self, surface: &str) -> Vec<usize> {
        match self.inner.vocab.get_id_by_surface(surface) {
            Some(id) => self.inner.vocab.get_root_set(id).to_vec(),
            None => Vec::new(),
        }
    }

    // ---------------- E2 induction  [EQ-6] ----------------

    /// Induced units enter as V_ambiguous ONLY, gated by c(u) >= c_lo. There is
    /// deliberately no path from here to V_strict.
    fn induce_ambiguous_seeds(
        &mut self,
        candidates: Vec<(String, f64)>,
        c_lo: f64,
    ) -> PyResult<usize> {
        let mut added = 0usize;
        for (surface, c) in candidates {
            if c >= c_lo && !surface.is_empty() {
                self.inner.vocab.add_induced_ambiguous(surface, c);
                added += 1;
            }
        }
        Ok(added)
    }

    /// Promotion V_ambiguous -> V_strict. Requires `human_validated=True`;
    /// c(u) alone can never promote.
    #[pyo3(signature = (surface, human_validated=false))]
    fn promote_to_strict(&mut self, surface: &str, human_validated: bool) -> bool {
        self.inner.vocab.promote_to_strict(surface, human_validated)
    }

    // ---------------- Phase 5  [EQ-4] ----------------

    /// L : surface -> RootID. Required for suppletion (गयो/जानु share nothing).
    fn set_lexicon(&mut self, lexicon: HashMap<String, usize>) {
        self.inner.lexicon = lexicon;
    }

    #[pyo3(signature = (seed_ambiguous_by_prior=false))]
    fn build_paradigm_embedding(&mut self, seed_ambiguous_by_prior: bool) -> usize {
        self.inner.build_paradigm_embedding(seed_ambiguous_by_prior)
    }

    /// U(t) per token id, -1 for ⊥. Feed straight into nn.Embedding.
    fn paradigm_embedding_ids(&self) -> Vec<i64> {
        self.inner.u_vector()
    }

    /// script(t) per token id: 0=DEV 1=LAT 2=PUN 3=FMT 4=MAL.
    fn script_ids(&self) -> Vec<u8> {
        self.inner.vocab.script_vector()
    }

    // ---------------- training ----------------

    fn train_bpe(
        &mut self,
        sequences: Vec<Vec<usize>>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        self.tag_roots();
        let mut corpus = Corpus::from_sequences(sequences, vocab_budget);
        let mut trainer = self.make_trainer();
        trainer.theta = theta;
        trainer.train(&mut corpus, vocab_budget, 0);
        self.reclaim(trainer);
        Ok(self.inner.vocab.len())
    }

    /// [EQ-7] Splits on whitespace and trains over word TYPES, so `same_word` is
    /// enforced on this path too. The old version handed whole texts to the
    /// trainer, which let merges cross word boundaries in violation of Freq's
    /// side condition.
    fn train_from_text(
        &mut self,
        texts: Vec<String>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        self.tag_roots();

        let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
        for t in &texts {
            let norm = self.inner.normalizer.normalize(t);
            for word in norm.split_whitespace() {
                let toks = self.inner.encode_normalized(word);
                if !toks.is_empty() {
                    *counts.entry(toks).or_insert(0) += 1;
                }
            }
        }
        if counts.is_empty() {
            return Ok(self.inner.vocab.len());
        }

        let mut corpus = Corpus::from_word_counts(counts, vocab_budget);
        let mut trainer = self.make_trainer();
        trainer.theta = theta;
        trainer.train(&mut corpus, vocab_budget, 0);
        self.reclaim(trainer);
        Ok(self.inner.vocab.len())
    }

    #[pyo3(signature = (path, vocab_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        vocab_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        let (counts, lines_done, word_occ, build_secs) =
            self.build_word_counts(py, &path, min_word_freq, progress_lines, None)?;
        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        self.tag_roots();
        let mut corpus = Corpus::from_word_counts(counts, vocab_budget);
        let mut trainer = self.make_trainer();
        trainer.theta = theta;
        let final_vocab = py.allow_threads(|| {
            trainer.train(&mut corpus, vocab_budget, progress_merges);
            trainer.vocab.len()
        });
        self.reclaim(trainer);
        Ok(final_vocab)
    }

    #[pyo3(signature = (path, dev_budget, lat_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_bilingual_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        dev_budget: usize,
        lat_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        let (counts, lines_done, word_occ, build_secs) =
            self.build_word_counts(py, &path, min_word_freq, progress_lines, None)?;
        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        self.tag_roots();
        let total_budget = dev_budget + lat_budget;
        let mut corpus = Corpus::from_word_counts(counts, total_budget);
        let mut trainer = self.make_trainer();
        trainer.theta = theta;

        let final_vocab = py.allow_threads(|| {
            trainer.latin_pass = false;
            trainer.train(&mut corpus, dev_budget, progress_merges);
            let after_dev = trainer.vocab.len();

            trainer.latin_pass = true;
            trainer.train(&mut corpus, total_budget, progress_merges);
            let after_lat = trainer.vocab.len();

            eprintln!(
                "[train] budget split: DEV {} (target {}) | LAT +{} (target +{})",
                after_dev,
                dev_budget,
                after_lat - after_dev,
                lat_budget
            );
            trainer.vocab.len()
        });
        self.reclaim(trainer);
        Ok(final_vocab)
    }

    #[pyo3(signature = (path, lat_budget, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_latin_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        lat_budget: usize,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        if self.inner.vocab.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "train_latin_from_file: vocab is empty — call load_vocab_tsv first",
            ));
        }
        let (counts, lines_done, word_occ, build_secs) =
            self.build_word_counts(py, &path, min_word_freq, progress_lines, Some(true))?;
        eprintln!(
            "[build:LAT] done: {} lines, {} latin-word-occ, {} unique types kept \
             (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        let target = self.inner.vocab.len() + lat_budget;
        let mut corpus = Corpus::from_word_counts(counts, target);
        let mut trainer = self.make_trainer();
        trainer.theta = 100;
        trainer.latin_pass = true;
        let final_vocab = py.allow_threads(|| {
            trainer.train(&mut corpus, target, progress_merges);
            trainer.vocab.len()
        });
        self.reclaim(trainer);
        Ok(final_vocab)
    }

    // ---------------- E4 metrics  [EQ-9] ----------------

    /// Intrinsic metrics computed by the SAME encode path the model will use,
    /// so eval can't silently drift from training.
    ///   fertility        = tokens / whitespace-word
    ///   tokens_per_char  = tokens / char of N(s)
    ///   byte_fallback_rate = MAL tokens / tokens
    ///   dev_budget_share = |{t : script(t)=DEV}| / |V|
    /// BPC is deliberately NOT here: it needs model log-probs. Compute it as
    /// (Σ -log2 p(token)) / (bytes of N(s)) with the byte count this returns.
    fn corpus_stats(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyDict>> {
        let mut tokens = 0u64;
        let mut words = 0u64;
        let mut chars = 0u64;
        let mut bytes = 0u64;
        let mut mal = 0u64;
        let mut roundtrip_ok = 0u64;

        for t in &texts {
            let norm = self.inner.normalizer.normalize(t);
            chars += norm.chars().count() as u64;
            bytes += norm.len() as u64;
            words += norm.split_whitespace().count() as u64;
            let ids = self.inner.encode_normalized(&norm);
            tokens += ids.len() as u64;
            for &id in &ids {
                if self.inner.vocab.get_script(id) == Script::MAL {
                    mal += 1;
                }
            }
            if self.inner.decode(&ids) == norm {
                roundtrip_ok += 1;
            }
        }

        let d = PyDict::new(py);
        d.set_item("texts", texts.len())?;
        d.set_item("tokens", tokens)?;
        d.set_item("words", words)?;
        d.set_item("chars", chars)?;
        d.set_item("bytes", bytes)?;
        d.set_item("fertility", tokens as f64 / (words.max(1) as f64))?;
        d.set_item("tokens_per_char", tokens as f64 / (chars.max(1) as f64))?;
        d.set_item("byte_fallback_rate", mal as f64 / (tokens.max(1) as f64))?;
        d.set_item("roundtrip_pass_rate", roundtrip_ok as f64 / (texts.len().max(1) as f64))?;
        d.set_item("vocab_size", self.inner.vocab.len())?;
        d.set_item(
            "dev_budget_share",
            self.inner.vocab.count_script(Script::DEV) as f64
                / (self.inner.vocab.len().max(1) as f64),
        )?;
        d.set_item(
            "lat_budget_share",
            self.inner.vocab.count_script(Script::LAT) as f64
                / (self.inner.vocab.len().max(1) as f64),
        )?;
        Ok(d.into())
    }

    // ---------------- persistence ----------------

    fn load_vocab(&mut self, pairs: Vec<(usize, String)>) -> PyResult<usize> {
        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    #[staticmethod]
    fn escape_tsv(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '\t' => out.push_str("\\t"),
                '\n' => out.push_str("\\n"),
                _ => out.push(ch),
            }
        }
        out
    }

    /// [EQ-5] Reads the `# nepbpe-v4 mode=…` header and REFUSES a vocabulary
    /// built under a different folding mode. This is the build-time fork made
    /// unbypassable: the ids of an N_LM vocab do not mean the same thing under
    /// N_OCR. Headerless (pre-v4) files are accepted with a warning.
    #[pyo3(signature = (path, allow_mode_mismatch=false))]
    fn load_vocab_tsv(&mut self, path: String, allow_mode_mismatch: bool) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);

        let mut pairs: Vec<(usize, String)> = Vec::new();
        let mut file_mode: Option<FoldMode> = None;

        for line in reader.lines() {
            let line = line.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read: {}", e))
            })?;
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                if let Some(idx) = rest.find("mode=") {
                    file_mode = FoldMode::from_tag(&rest[idx + 5..]);
                }
                continue;
            }
            let mut it = line.splitn(2, '\t');
            let id_str = match it.next() {
                Some(s) => s,
                None => continue,
            };
            let surf_raw = it.next().unwrap_or("");
            let id: usize = match id_str.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            pairs.push((id, unescape_tsv(surf_raw)));
        }

        match file_mode {
            Some(fm) if fm != self.inner.normalizer.mode() && !allow_mode_mismatch => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "folding-mode mismatch: vocab was built with mode={} but this \
                     tokenizer is mode={}. Modes are a build-time fork — one \
                     tokenizer, one vocabulary, one model per mode. Rebuild, or \
                     pass allow_mode_mismatch=True if you know what you are doing.",
                    fm.tag(),
                    self.inner.normalizer.mode().tag()
                )));
            }
            None => {
                eprintln!(
                    "[load] warning: {} has no mode header; assuming mode={}",
                    path,
                    self.inner.normalizer.mode().tag()
                );
            }
            _ => {}
        }

        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    fn save_vocab_tsv(&self, path: String) -> PyResult<()> {
        use std::fs::File;
        use std::io::Write;

        let mut pairs = self.inner.vocab.get_all_surfaces();
        pairs.sort_by_key(|(id, _)| *id);

        let f = File::create(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("create {}: {}", path, e))
        })?;
        let mut writer = std::io::BufWriter::new(f);

        writeln!(
            writer,
            "# nepbpe-v4 mode={}",
            self.inner.normalizer.mode().tag()
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("write: {}", e)))?;

        for (id, surf) in pairs {
            let escaped = Self::escape_tsv(&surf);
            writeln!(writer, "{}\t{}", id, escaped).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("write: {}", e))
            })?;
        }
        Ok(())
    }

    fn get_token_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }

    fn get_vocab_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (id, surf) in self.inner.vocab.get_all_surfaces() {
            dict.set_item(surf, id)?;
        }
        Ok(dict.into())
    }

    fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        let ids = self.inner.encode(text);
        ids.iter()
            .filter_map(|&id| self.inner.vocab.get_surface(id))
            .collect()
    }
}

impl PyHimalayanTokenization {
    /// Shared streaming word-count builder. `latin_only=Some(true)` applies the
    /// ASCII-alphanumeric filter that keeps the dictionary English-sized.
    fn build_word_counts(
        &self,
        py: Python<'_>,
        path: &str,
        min_word_freq: u64,
        progress_lines: u64,
        latin_only: Option<bool>,
    ) -> PyResult<(HashMap<Vec<TokenId>, Frequency>, u64, u64, f64)> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        let file = File::open(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;
        let lat_only = latin_only.unwrap_or(false);

        Ok(py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    if lat_only && !word.chars().any(|c| c.is_ascii_alphanumeric()) {
                        continue;
                    }
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build] {} lines | {} word-occ | {} unique | {:.1}s",
                        lines_done,
                        word_occ,
                        counts.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        }))
    }
}

#[pymodule(name = "HimalayanTokenization")]
fn himalayan_tok(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHimalayanTokenization>()?;
    Ok(())
}