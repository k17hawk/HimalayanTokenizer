# HimalayanTokenizer v4 — Complete Equation Set

v4 = v2 (corrected) + Part II extensions **in their implemented form**. The change
from v3 is not new theory: it is that nine equations which v2/v3 stated but the
reference implementation never executed are now executed, and three of them had to
be restated to be true of a running system.

Notation as v2. Additions: `μ` = marker selector, `M` = boundary marking,
`Units` = the unit alphabet of a run, `≺_lex` = lexicographic order.

---

## Phase 1 — Normalization `N_m` (mode-indexed, build-time fork)

```
N_LM(s)  = NFC ∘ StripZWJ ∘ Fold_O^full    ∘ NFC (s)
N_OCR(s) = NFC ∘ KeepZWJ  ∘ Fold_O^minimal ∘ NFC (s)
```

ZWNJ (U+200C) is preserved in **both** modes as `τ_ZWNJ`. The two modes differ in
(i) the folding table and (ii) whether ZWJ (U+200D) survives.

> **Change from v2/v3.** `N` is now `NFC`-**closed on both ends**. v2 applied NFC only
> before folding, which made idempotence conditional on every replacement being
> NFC-stable — a caveat, not a guarantee. The trailing `NFC` discharges it:

```
N_m(N_m(s)) = N_m(s)        for all s, unconditionally on replacement stability
```

The one remaining obligation is that no replacement re-triggers a pattern. This is
now **checked**, not assumed:

```
Idempotent(Fold_O)  ⟺  ∀ (p→r) ∈ Fold_O, ∀ (q→·) ∈ Fold_O :  q ⊄ r
```

`validate_folding_rules()` returns the violating pairs; empty ⇒ the condition holds.

**Losslessness order:** `N_OCR ≺ N_LM` strictly (`N_OCR` distinguishes सँग / संग /
सङ्ग and keeps ZWJ). Mode is frozen into the vocabulary; the vocab file carries
`# HimalyanTokenizer-v4 mode=…` and loading under the other mode is a hard error.

---

## Phase 1.5 — Boundary marking `M` (new; was in the code, never in the equations)

This phase existed only as an implementation detail and was the source of the one
remaining roundtrip hole. Stated properly:

```
μ(c) = ▂   if c ∈ [A-Za-z0-9]
     = ▁   otherwise (including c = ⊥, i.e. end of input)

M(s) = ⟨μ(s₁), true⟩ · ⨁_{i=1..|s|}  ⟨μ(s_{i+1}), true⟩   if sᵢ = ' '
                                     ⟨sᵢ,        false⟩   otherwise
```

`M` produces a sequence of **tagged** characters `⟨c, synthetic⟩`. The tag is
load-bearing:

```
script(⟨▁,true⟩)  = DEV
script(⟨▂,true⟩)  = LAT
script(⟨c,false⟩) = MAL          if c ∈ {▁, ▂}     # literal-marker escape
                  = script(c)    otherwise
```

> **Fix #10 (marker aliasing).** Without the escape clause, a literal U+2581 in the
> input is indistinguishable from an inserted word-start marker, and `M⁻¹` turns it
> into a space. `Decode(Encode(s)) = N(s)` was therefore false for any `s` containing
> ▁ or ▂. Routing literal markers to `MAL` sends them through byte-fallback, where
> `M⁻¹` does not touch them.

```
M⁻¹(t₁…t_k) = StripOneLeadingSpace ( ⨁_i  unmark(tᵢ) )

unmark(t) = bytes(t)                                     if script(t) = MAL
          = surface(t)[▁ ↦ ' ', ▂ ↦ ' ']                 otherwise
```

> **Fix #11 (unmark by type, not by string).** v2's implementation did a global
> string replace on the fully decoded output, which cannot distinguish a marker that
> arrived through a token surface from one reconstructed out of bytes. `M⁻¹` is now
> defined per token, keyed on `script(t)`.

```
M⁻¹(M(s)) = s        exact, for all s
```

Runs never cross a **synthetic** marker (that would fuse two words); literal markers
break runs on script alone.

### Roundtrip guarantee

```
Decode(Encode_m(s)) = N_m(s)        exact, for ALL byte inputs, all m
```

---

## Phase 2 — Akshara DFA, tagging, byte-fallback

### 2.0 The DFA `D` (now concrete, and now actually used)

```
Q = { q₀ start, q_C consonant, q_H half-form, q_V vowel, q_N nasal, q_A atomic }
F = { q_C, q_H, q_V, q_N, q_A }                      # q_H ∈ F: क् is a complete akshara

δ:  q₀ --C--> q_C          q₀ --V--> q_V          q₀ --digit|ॐ|ऽ--> q_A
    q_C --nukta--> q_C     q_C --virama--> q_H    q_C --matra--> q_V    q_C --nasal--> q_N
    q_H --C--> q_C                                                       # conjunct: C ् C
    q_V --matra--> q_V     q_V --nasal--> q_N

Akshara(x) = maximal munch on δ with backtracking to the last state in F
```

`q_H ∈ F` is what makes a word-final half consonant atomic while still letting
`क ् ष` bind into `क्ष` — the longer accepting run wins.

> **Fix #12 (inert DFA).** v2 shipped `D = ∅`, so `Akshara(x)` degenerated to one
> character per unit and **every "exact" conjunct guarantee was vacuous**. `D` is now
> populated by default over U+0900–U+097F.

### 2.1 `script(·)` — total

```
script(t) = DEV | LAT | PUN | FMT (τ_ZWNJ) | MAL (β_x)
```

`FMT` and `MAL` are atomic classes: never initiate, never continue a merge.

### 2.2 Vocabulary

```
V₀ = BaseAksharas ∪ V_seed ∪ Σ_P ∪ Σ_P^ext ∪ Σ_LAT ∪ { ▁, ▂ } ∪ { β_x : x∈[0..255] } ∪ { τ_ZWNJ }
V_seed = V_strict ∪ V_ambiguous                       # seeded, never merge-built
Σ_P ⊇ { । (U+0964), ॥ (U+0965) }
```

`BaseAksharas` is obtained by running `D` over the corpus (`harvest_aksharas`), so
the inventory is guaranteed to cover it.

### 2.3 Root tagging

```
RootSet(α) = { root : α ∈ prefixes(P(root).states) },  sorted by descending P(root)
```

> **Fix #13.** The sort is not cosmetic: `root_set_cap` in E1 truncates `RootSet`, and
> truncating an unsorted set discards mass arbitrarily. Sorted-by-prior makes the cap
> a principled top-mass approximation.

---

## Phase 3 — Constrained BPE

### 3.1 Frequency

```
Freq(a,b) = Σ_{w ∈ WordTypes} count(w) · |{ i : w[i]=a ∧ w[i+1]=b }|
```

> **Fix #14 (`same_word` made structural).** `same_word(i,i+1)` is no longer a side
> condition to be honoured — the corpus **is** a bag of word types, so `windows(2)`
> cannot straddle a boundary. This closes the in-memory training path, which
> previously fed whole texts to the trainer and let merges cross words.

### 3.2 Admissibility — `Legal = Gate ∧ Morph` (UNCHANGED by E1)

```
ScriptCompat(a,b) = script(a)=script(b) ∧ script(a) ∈ {DEV, PUN}      # Phase 3
                  = script(a)=script(b) ∧ script(a) = LAT             # Phase 4

Gate(a,b) =
  0   if script(a) ∈ {MAL,FMT} ∨ script(b) ∈ {MAL,FMT}
  0   if ¬ScriptCompat(a,b)
  0   if b ∈ V_seed
  0   if a ∈ V_strict
  0   if a ∈ V_ambiguous ∧ Freq(a,b) < θ                        θ = 100
  1   otherwise

Morph(a,b) =
  1   if RootSet(a) = ∅
  1   if ∃ root ∈ RootSet(a) : b ∈ P(root).states[a].allowed_next
  0   otherwise
```

Existential and **hard**. E1 does not touch it. Narrowing:

```
RootSet(a∘b) = { root ∈ RootSet(a) : b ∈ P(root).states[a].allowed_next }   if RootSet(a)≠∅
             = ∅                                                            otherwise
```

### 3.3 Priority `K` — full form (E1 folded in)

```
P(root | hist(a)) = P(root) · 1[root ∈ RootSet(a)] / Σ_{r ∈ RootSet(a)} P(r)

MorphScore(a,b) = 1                                                    if RootSet(a) = ∅
                = 1                                                    if Σ_{r} P(r) = 0     # degenerate prior
                = Σ_{root ∈ RootSet(a)} P(root|hist(a)) · 1[b ∈ allowed_next]   ∈ [0,1]

ScriptRank(DEV)=2,  ScriptRank(PUN)=1,  ScriptRank(LAT)=0
W(a) = intra-tier weight, default 1

K(a,b) = ( ScriptRank(script(a)),  ⌊ MorphScore(a,b) · W(a) · Freq(a,b) · 2¹⁰ ⌋ )

merge* = argmax_{(a,b) ∈ A} K(a,b)      under ≺_lex
```

> **Fix #15 (fixed-point key).** `MorphScore·W·Freq` is real-valued; a heap ordered on
> floats is not reproducible across builds and cannot be a total order with a
> deterministic tie-break. Scaling by `2¹⁰` and flooring gives an exact `u64` key.
> Headroom: `Freq ≤ 10¹²` ⇒ key `≤ 10¹⁵ ≪ 2⁶⁴`.
>
> **Degeneracy clauses.** Both `RootSet = ∅` and `Σ P(r) = 0` return 1, so a missing or
> all-zero prior degrades E1 to v2's `K` rather than zeroing out admissible merges.
> **E1 off ⇒ `MorphScore = W = 1` ⇒ `K` is v2's `K` exactly**, so the extension is a
> strict superset with an ablation switch (E4 requires this).

Ties broken by `(a, b)` ascending — trained vocabularies are bit-reproducible.

### 3.4 Selection — lazy heap invalidation

```
Heap entry := ( K(a,b), a, b, freq_snapshot )                 max-heap on K, ≺_lex

On applying merge* = (a,b):
    for each touched corpus position:  decrement old adjacent pairs,
                                       form (x, a∘b) and (a∘b, y),
                                       compute RootSet(a∘b), re-evaluate Legal,
                                       push a fresh entry for EVERY affected pair
                                       (including the decremented neighbours)
On pop:
    discard if Freq(a,b) ≠ freq_snapshot         # stale
    discard if Freq(a,b) = 0                     # consumed
    discard if Legal(a,b) = 0                    # admissibility revoked
    otherwise it is the current argmax
```

Re-pushing decremented neighbours is required: their snapshots are stale, so without
a fresh entry a still-legal pair vanishes silently.

### 3.5 Checkpoint / resume

Training state after `n` merges:

```
Σ_n = ( V_n , R_n , C_n , Θ )

V_n = vocabulary: surfaces, variants, script(·), V_strict, V_ambiguous, c(u)
R_n = RootSet_n : TokenID → 2^RootID
C_n = { (w, count(w)) } with w the CURRENT token sequence of each word type
Θ   = ( θ, latin_pass, P(root), W, E1 flags, mode m, L, dev_budget, total_budget )
```

**Derived — deliberately not serialized:**

```
Freq_n = Σ_{w ∈ C_n} count(w) · |{i : w[i]=a ∧ w[i+1]=b}|      # refold over C_n
A_n    = { (a,b) adjacent in C_n : Legal(a,b) = 1 }
Heap_n = build(A_n)                                            # rebuilt from A_n
```

**Resume exactness.** Let `merge*_{n+1}` be the merge chosen by an uninterrupted run
and `merge*'_{n+1}` the one chosen after restoring `Σ_n` and rebuilding the heap.
Then `merge*'_{n+1} = merge*_{n+1}`.

> *Proof.* By §3.4's invariant, the first popped entry surviving both checks is
> `argmax_{(a,b)∈A_n} K(a,b)` under the current corpus state. A rebuilt heap contains
> exactly `A_n`, each with `K` evaluated at the current state and no stale entries, so
> its first pop is that same argmax. `K` is a pure deterministic function of `Σ_n`
> (§3.3, fixed-point), and ties break totally on `(a,b)`. Since `create_merged` assigns
> ids sequentially in merge order, id assignment matches too. Induction on `n` gives
> `V_final` identical to the uninterrupted run. ∎

Rebuilding is therefore not an approximation — it is *strictly safer* than
serializing `Heap_n`, which would persist stale entries whose `freq_snapshot` no
longer means anything after a restore.

**Two components that look derivable and are not:**

```
R_n ≠ RootSet-from-registry(V_n)
    narrowing is path-dependent: RootSet(a∘b) is a function of the merge SEQUENCE.
    Re-running the §2.3 tagging on resume overwrites narrowed sets with
    un-narrowed ones and monotonically LOOSENS Morph. The resume path must not
    call it.

C_n ≠ Encode_{V_n}(corpus)
    greedy-longest-match over V_n is a different partition than applying merges
    1..n in order. C_n is genuine state, not a cache.
```

**Commit protocol** (crash cannot yield a valid-looking partial state):

```
write V_n, C_n, Θ  →  write COMPLETE  →  replace LATEST

Valid(ckpt) ⟺ COMPLETE ∈ ckpt
Resolve(root) = the LATEST-named step if Valid, else max{ step : Valid(step) }
```

Saves are taken **between** merges — after `merge*` has been applied to both `V` and
`C`, before the next pop — so `Σ` is always consistent at an integer `n`, never a
vocabulary one merge ahead of its corpus.

**Resume guards.** `P` and the corpus are build *inputs*, not state, so they are
fingerprinted rather than serialized:

```
fp(C₀)  = ⊕_w FNV1a( w ‖ count(w) )          # commutative: HashMap order-independent
fp(P)   = ⊕_{(root, prefix, cont)} FNV1a(·)

Resume(Σ_n) requires:   m_ckpt = m_now  ∧  fp(P)_ckpt = fp(P)_now
```

`fp(C₀)` is computed once at build and carried forward unchanged (`C_n` hashes
differently by construction), so it identifies the corpus a checkpoint belongs to.
Mode mismatch and paradigm mismatch are hard errors — resuming across either changes
what the ids mean or what `Morph` admits, mid-run.

**Phase resumption.** `Θ` records `phase ∈ {DEV, LAT}` plus both budgets, so a
bilingual run resumes correctly from either pass:

```
phase = DEV →  train_DEV to dev_budget ; then train_LAT to total_budget
phase = LAT →  train_LAT to total_budget
```

An unconditional end-of-phase save makes the DEV→LAT handoff itself a resume point.

**Size.** `|C_n|` shrinks monotonically in `n` (every merge removes one token from
every occurrence it consumes), so checkpoints get cheaper as training proceeds; the
first one is the worst case.

---

## Phase 4 — Latin secondary pass

Unconstrained BPE over `LAT` runs only, filling `B − |V|`. `Gate` reduces to
`ScriptCompat_LAT`; `Morph ≡ 1`. `▁` is DEV and `▂` is LAT, so `▂the` folds into one
token in this pass while `▁माया` folds in Phase 3, and no token spans DEV/LAT.

---

## Encode — the guarantee-carrying definition

This is the equation v2 never wrote down, and its absence is why the conjunct
guarantee did not hold in practice.

```
Units(run) =  ⟨marker⟩ · Akshara(run∖marker)      if script(run) = DEV
           =  per-character                       if script(run) = LAT
           =  per-character                       if script(run) = PUN

Encode(run) = greedy-longest-match over CONCATENATIONS OF WHOLE UNITS:
    at position i, emit argmax_{ j > i, |units[i..j]| ≤ maxlen } { t ∈ V : surface(t) = ⨁ units[i..j] }
    if no such t:  emit β-spelling of units[i] entirely, advance by ONE UNIT
```

> **Fix #16 (the big one).** v2's encoder ran greedy match at **character**
> granularity over DEV runs. A learned merge could therefore emit a token ending
> mid-akshara — the exact failure Phase 2 was designed to prevent. Because merges are
> now built from akshara units *and* matched over akshara units, token boundaries are
> a subset of akshara boundaries at both training and inference:

```
Boundaries(Encode(s)) ⊆ Boundaries(Akshara(N(s)))
```

An out-of-vocabulary akshara is spelled out in byte-fallback **as a whole unit**, so
even the fallback path emits no DEV token straddling a matra or conjunct.

---

## Phase 5 — Model integration

```
Input(t) = TokenEmb(t) + PosEmb(pos) + ScriptEmb(script(t)) + ParadigmEmb(U(t))
```

```
U : TokenID → RootID ∪ {⊥}          many-to-one, seeded, then LEARNED

U(t) = L(surface(t))                        if surface(t) ∈ dom(L)     # 1. lexicon wins
     = r                                     if RootSet(t) = {r}       # 2. narrowing resolved it
     = argmax_{r ∈ RootSet(t)} P(r)          if |RootSet(t)| > 1 ∧ seed_by_prior
     = ⊥                                     otherwise
```

Ordering matters. `L` is checked **first** because it is the only clause that can
express suppletion (हुनु → भयो / छ / थियो, जानु → गयो share no substring with the
root, so no form-based clause can reach them). Clause 3 is off by default: a
prior-argmax seed on an unresolved `RootSet` is a guess, and E4's ParadigmEmb
ablation needs `{off, random-init, P-seeded}` to be clean.

> **Fix #17.** v2 declared `U` and then never built it — the field existed, was never
> populated, and was never exported. `U(t)` and `script(t)` are now emitted as
> id-aligned integer vectors (`⊥ ↦ −1`) so `ScriptEmb` / `ParadigmEmb` have real
> inputs and the headline ablation is runnable.

---

## E2 — Seed induction (augment, never replace)

```
V_strict    = Curated_closed_class                          # hand-validated; only door
V_ambiguous = { u ∈ Induced : c(u) ≥ c_lo }  ∪  Curated_ambiguous

Promotion:  V_ambiguous → V_strict  requires human_validated = true
            c(u) alone NEVER promotes
```

Induced units are still subject to `θ` via the `a ∈ V_ambiguous ∧ Freq < θ` Gate
clause, so induction affects vocabulary **size**, never the hard-frozen core.

---

## E5 — FST behind `P(root)` (keystone, unchanged)

```
FST generation  →  P(root).states[a].allowed_next          # Phase 3 constraint
FST analysis    →  surface → root + features
                   ├─ seeds L and U                         # Phase 5
                   └─ yields P(root) prior                   # E1
```

Determinized + minimized to keep `O(1)` transitions. Suppletion is a **lexical
stipulation** inside the FST lexicon — listed, not derived; listing is all that is
ever possible. Ambiguous analysis returns a **parse lattice**, which surfaces the
ambiguity rather than resolving it.

---

## E4 — Evaluation protocol

```
CONTROL-1  Matched vocabulary budget across ALL tokenizers
CONTROL-2  Report BPC / bits-per-byte, NOT perplexity
```

Metrics computed **through the production encode path**, so eval cannot drift from
training:

```
fertility          = tokens / whitespace-word
tokens_per_char    = tokens / |N(s)|_chars
byte_fallback_rate = |{t : script(t) = MAL}| / tokens
dev_budget_share   = |{t ∈ V : script(t) = DEV}| / |V|
roundtrip_pass_rate= |{s : Decode(Encode(s)) = N(s)}| / |S|          # should be 1.000
BPC                = ( Σ_t −log₂ p(t) ) / |N(s)|_bytes               # model-side
```

Baselines at matched vocab: byte-BPE · SentencePiece-BPE · SP-Unigram · Morfessor ·
an existing Indic tokenizer · HimalyanTokenizer. Ablations: `ParadigmEmb {off, random, P-seeded}`
(headline), `Morph {existential-hard, +probabilistic-K, off}`, `Folding {N_LM, N_OCR}`.

---

## What v4 Guarantees

**Exact, by construction:**

| Guarantee | Mechanism | v2 status |
|---|---|---|
| No token splits a matra or breaks a conjunct | `D` populated + unit-granular greedy match | **claimed but false** — DFA empty, char-level match |
| Zero tokens span two scripts | `ScriptCompat` | held |
| `Decode(Encode_m(s)) = N_m(s)` for all bytes | β-fallback + literal-marker escape + per-token `M⁻¹` | **false for inputs containing ▁/▂** |
| `N_m(N_m(s)) = N_m(s)` | trailing NFC + rule validator | conditional |
| `V_strict` frozen and terminal; `V_ambiguous` θ-gated | Gate | held |
| Strict script dominance in selection | lexicographic `K` | held |
| Freq respects `same_word` | corpus = bag of word types | held on file path, **violated on in-memory path** |
| Merge order reproducible bit-for-bit | fixed-point `K` + `(a,b)` tie-break | held (no `MorphScore` to break it) |
| Folding mode cannot be silently mixed | mode header + load-time refusal | not enforced |
| Resume produces the identical vocabulary | §3.5 exactness proof; heap rebuilt, not restored | **no checkpointing at all** |
| A crashed checkpoint write is never loaded | `COMPLETE` sentinel written last | n/a |

**Sound but data-dependent in tightness:** no merge unlicensed by *all* still-consistent
roots (`Morph`) — loose while `|RootSet|` is large, tightens with narrowing. E1
reorders within this set; it does **not** tighten it, by design.

**Data-dependent:** long-tail morphology quality; allomorph/suppletion unification via
`ParadigmEmb` + `L`.

**Provably out of scope:** unifying non-contiguous surfaces (delegated to `U`);
recovering sandhi-erased boundaries (विद्या + आलय → विद्यालय); context-dependent
segmentation (`Score` has zero sentence context — needs a lattice handed to the model).

---

## Complexity

```
Gate                 : O(1)
Morph                : O(|RootSet(a)|) → O(1) after narrowing
MorphScore (E1)      : O(min(|RootSet(a)|, cap))
Selection            : O(log|A|) amortized, lazy invalidation
Checkpoint save      : O(|V| + Σ|w|)  ~ one linear pass, shrinks with n
Resume               : O(|V| + Σ|w| + |A| log|A|)   incl. one heap rebuild
Akshara segmentation : O(|run| · maxAksharaLen), maxAksharaLen small and bounded
Unit greedy match    : O(maxlen) surface lookups per position
```

The `O(log|A|)` bound holds **only** under §3.4's lazy invalidation. A fixed-`K` heap
is incorrect, not merely slow.

---

## Pre-compute gates (unchanged, all still binding)

1. BPC-not-PPL + matched vocab — correctness of the entire eval.
2. Train-time freeze is irreversible **per model**: every `P` / FST / folding decision
   is a pre-compute gate. The tokenizer cannot be patched post-hoc.
3. FST coverage number on a real corpus **before** elegance matters. An FST analyzing
   60% of tokens leaves 40% on the loose existential path, and no probabilistic
   ranking saves that.

New in v4:

4. **`byte_fallback_rate` and `roundtrip_pass_rate` on a held-out corpus before any
   training run.** These are cheap, and they are the two numbers that catch a broken
   akshara inventory or a marker-escaping regression before compute is spent.
5. **Resume-equivalence test before the real run.** Train to `n` merges, checkpoint,
   kill, resume to `2n`; separately train straight through to `2n`; assert the two
   vocabularies are identical id-for-id. §3.5 says this must hold — if it does not,
   something in `Σ` is not being persisted, and the failure will otherwise only
   surface as a quietly degraded vocabulary days into a run.



# HimalayanTOK — Nepali

An akshara-aware, script-tiered BPE tokenizer for Nepali, written in Rust with Python
bindings. Perfect Devanagari coverage, zero UNK, and a hard guarantee that no token
ever splits a matra from its consonant or breaks a conjunct.

**Status: working resource tokenizer. The morphology layer is built but not yet
activated — see [Known Limitations](#known-limitations) before you cite anything.**

---

## What it is

Most multilingual tokenizers treat Nepali as a byte stream that happens to be
Devanagari. They split `काठमाडौं` mid-conjunct, strip the matra off its consonant, and
spend four tokens on a word that should cost one. HimalayanTOK constrains BPE so those
failures are impossible by construction rather than unlikely in practice.

Four phases:

| Phase | What it does |
|---|---|
| 1. Normalization `N` | NFC → orthographic folding (सँग/संग/सङ्ग → one form) → ZWJ handling. ZWNJ preserved as an explicit token. |
| 2. Akshara DFA | Segments Devanagari into well-formed aksharas. Matras bind to their consonant; conjuncts stay whole. Malformed bytes get a 256-token byte-fallback alphabet. |
| 3. Constrained BPE | Merges are admissible only if they pass a script gate and a paradigm check. Priority is **lexicographic**: any Devanagari merge outranks any punctuation merge regardless of frequency. |
| 4. Latin pass | Separate unconstrained BPE over Latin runs. No token can span two scripts. |

The full equation set — including what the design provably *cannot* do — is in
[`NepBPE_v4_equations.md`](./NepBPE_v4_equations.md).

---

## Benchmark

Nepali evaluation corpus, seven tokenizers, off-the-shelf.

| Tokenizer | Vocab | Tok/Wrd | Bytes/Tok | UNK% | Deva% | Speed | Morph.Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HimalayanTOK-100K** | 100,001 | **1.606** | 6.97 | 0.00% | **100.0%** | 84,362/s | **0.000** |
| IndicBERT | 200,000 | 1.827 | 6.13 | 0.95% | 64.1% | 15,252/s | 1.000 |
| mBERT | 119,547 | 2.026 | 5.53 | 0.10% | 67.2% | 24,306/s | 0.727 |
| XLM-R | 250,002 | 1.617 | 6.93 | 0.00% | 61.7% | 16,311/s | 0.233 |
| TikToken-cl100k | 100,277 | 3.465 | 3.23 | 0.00% | 13.3% | 97,916/s | 0.686 |
| TikToken-o200k | 200,019 | 1.704 | 6.57 | 0.00% | 61.7% | 129,169/s | 0.727 |
| Llama-2 | 32,000 | 3.819 | 2.93 | 0.00% | 30.5% | 27,777/s | 0.648 |

Segmentation of `मेरो नाम राम हो र म काठमाडौंमा बस्छु`:

```
HimalayanTOK     मेरो | नाम | राम | हो | र | म | काठमाडौंमा | बस्छु
XLM-R            मेरो | नाम | राम | हो | र | म | काठमाडौंमा | बस | ्छु      ← orphan halant
IndicBERT        मर | नम | रम | ह | र | म | कठ | म | डम | बस | छ           ← matras dropped entirely
Llama-2          म|े|र|ो| |न|ा|म| |र|ा|म| ... (43 tokens)
```

### Read the caveats before quoting the table

- **Vocabulary budgets are not matched.** Fertility improves mechanically with vocab
  size. HimalayanTOK's 1.606 tok/word at 100K against XLM-R's 1.617 at 250K is
  suggestive, not a result. A publishable comparison requires retraining every
  baseline at the same budget on the same corpus.
- **Speed is not an algorithmic claim.** It's Rust against Python wrappers. (The
  benchmark output labels this "C++" — that's a mislabel; the implementation is Rust.)
- **There is no BPC number here.** Tokens/word is a proxy. The real question — does a
  language model trained on these tokens compress Nepali better — needs bits-per-character
  from an actual trained LM. Perplexity will not do; token inventories differ, so PPL
  comparisons across tokenizers are meaningless.

---

## Known limitations

### The morphology layer is inactive

`Morph.Share = 0.000` — the worst score in the benchmark. HimalayanTOK shares **zero**
tokens between paired verb conjugations where every other tokenizer shares at least
some:

| Pair | HimalayanTOK | IndicBERT | mBERT |
|---|---|---|---|
| जान्छु / जान्छौ | 0 shared (1, 13 tokens) | 2 shared | 2 shared |
| खान्छु / खान्छौ | 0 shared (1, 13 tokens) | 3 shared | 2 shared |
| गर्छु / गर्छौ | 0 shared (1, 1 tokens) | 2 shared | 3 shared |

This is not a subtle tuning problem. Two things are visibly wrong:

1. **The paradigm machinery is unpopulated.** `Morph` has a `RootSet(a) = ∅ ⇒ 1`
   clause, which is correct — it stops paradigm licensing from blocking every merge of
   every ordinary token. But with no FST loaded, *every* token has an empty RootSet, so
   the constraint is never exercised and the tokenizer is behaving as plain frequency
   BPE with a script gate. The morphology is architecture, not behavior.

2. **The 13-token forms are byte fallback firing.** `जान्छु` costs 1 token and `जान्छौ`
   costs 13. A six-character word exploding to 13 tokens means the encoder found no
   vocabulary entry and spelled it out byte by byte. The frequent inflection was
   memorized whole; the less frequent one fell off a cliff. That's the exact failure
   mode morphological segmentation is supposed to prevent, and it shows the base
   akshara inventory does not cover the corpus.

**What fixes it:** the FST (E5 in the roadmap). One resource populates the Phase-3
paradigm constraint, seeds the Phase-5 `U` map, and supplies E1's root prior. Until
it exists, describe this project as a fast frequency-BPE resource tokenizer with strong
Devanagari coverage — not as a morphological tokenizer.

### Other things to know

- **Vocabulary is 100,001, not 64K.** If you see "64K" anywhere in older docs, it's stale.
- **Byte-fallback rate is the number to watch.** Run `harvest_aksharas` before training
  so the base akshara inventory actually covers your corpus. An unseen akshara is spelled
  out in bytes — correct, but expensive.
- **The tokenizer cannot be patched after pretraining.** Folding mode, paradigm table,
  and vocabulary are frozen into token IDs. Every one of those is a pre-compute gate.
- **Loaded vocabularies are encode/decode-ready, not training-ready.** `V_strict`,
  `V_ambiguous`, and root sets are not restored from a TSV.

---

## Install

```bash
git clone https://github.com/<you>/HimalayanTOK-Nepali
cd HimalayanTOK-Nepali
pip install maturin
maturin develop --release
```

Requires Rust 1.70+ and Python 3.8+.

---

## Quickstart

### Train

```python
from HimalayanTOK_Nepali import PyHimalayanTOK_Nepali

tok = PyHimalayanTOK_Nepali(
    folding_rules=[("सङ्ग", "संग"), ("सँग", "संग")],
    mode="LM",                      # build-time fork: "LM" or "OCR", never a runtime flag
)

# Check N is idempotent for your rule set — empty list means it is.
assert tok.validate_folding_rules() == []

# Harvest the akshara inventory the DFA actually produces on YOUR corpus.
# Skipping this is the main cause of byte-fallback blowups.
aksharas = tok.harvest_aksharas("corpus.txt", min_freq=5)

tok.initialize_vocab(
    aksharas=aksharas,
    seed_morphemes=["हरू", "गा.वि.स."],
    punctuation=[".", ",", "।", "॥", "?", "!"],
    v_strict=["हरू"],               # unconditionally frozen, terminal
    v_ambiguous=[],                  # frequency-gated at θ
)

tok.train_bilingual_from_file(
    "corpus.txt",
    dev_budget=40_000,
    lat_budget=8_000,
    theta=100,
)

tok.save_vocab_tsv("vocab.tsv")     # stamped with mode=LM
```

### Encode

```python
tok = PyHimalayanTOK_Nepali(mode="LM")
tok.load_vocab_tsv("vocab.tsv")     # refuses a vocab built under a different mode

ids = tok.encode("मेरो नाम राम हो।")
print(tok.decode(ids))              # == tok.normalize(original)
assert tok.verify_roundtrip("मेरो नाम राम हो।")
```

### Evaluate

```python
stats = tok.corpus_stats(open("heldout.txt").read().splitlines())
print(stats["fertility"])            # tokens per whitespace word
print(stats["byte_fallback_rate"])   # should be near zero
print(stats["roundtrip_pass_rate"])  # should be exactly 1.0
```

`corpus_stats` runs through the production encode path, so evaluation cannot drift
from training. BPC is deliberately absent — it needs model log-probs. Compute it as
`(Σ −log₂ p(token)) / bytes`, using the byte count this returns.

### Morphology (once you have an FST)

```python
tok.add_paradigm(root_id=0, transitions={"जा": ["न्छु", "न्छौ", "न्छ"]})
tok.set_root_prior({0: 0.31, 1: 0.12})       # from FST analysis over the corpus
tok.enable_probabilistic_k(True, root_set_cap=16)
tok.assign_initial_roots()

# Phase 5: U : TokenID -> RootID. Lexicon first — it's the only thing that can
# express suppletion (गयो shares no substring with जानु).
tok.set_lexicon({"गयो": 0, "भयो": 1})
tok.build_paradigm_embedding()

u_ids = tok.paradigm_embedding_ids()          # -1 = ⊥, feed to nn.Embedding
script_ids = tok.script_ids()                 # 0=DEV 1=LAT 2=PUN 3=FMT 4=MAL
```

---

## Guarantees

Exact, by construction:

- No token splits a matra from its consonant or breaks a conjunct — token boundaries
  are a subset of akshara boundaries at both training and inference.
- Zero tokens span more than one script.
- `Decode(Encode(s)) = N(s)` for **all** byte inputs, garbage included.
- `N(N(s)) = N(s)`.
- `V_strict` morphemes are frozen and terminal; `V_ambiguous` extends only above θ.
- Devanagari always outranks punctuation in merge selection, regardless of frequency.
- Merge order is bit-reproducible across runs.

Sound but data-dependent: no merge is licensed by zero consistent roots. The filter is
loose while `RootSet` is large and tightens as narrowing disambiguates.

Provably out of scope — no equation of this family can do these:

- Unifying non-contiguous surfaces (आयो ↔ आउँछु, हुनु → भयो). Delegated to the learned
  `U` map plus a lexicon.
- Recovering boundaries erased by sandhi (विद्या + आलय → विद्यालय). The information is
  destroyed in the surface form.
- Context-dependent segmentation. Scoring has zero sentence context by construction;
  representing ambiguity needs a lattice handed to the model, a different object entirely.

---

## Roadmap

| | Item | Status |
|---|---|---|
| **E5** | FST behind `P(root)` — foma/HFST, determinized + minimized. **The keystone.** One resource serves the Phase-3 constraint, the Phase-5 `U` seeding, and E1's prior. | **Blocking everything below** |
| E1 | Probabilistic RootSet in the priority key. Implemented, inert without E5's prior. | Waiting on E5 |
| E2 | Semi-automatic seed induction. Induced units enter as `V_ambiguous` only; promotion to `V_strict` needs human sign-off. | Implemented |
| E3 | Folding modes as a build-time fork (`N_LM` / `N_OCR`), stamped into the vocab. | Implemented |
| E4 | Evaluation protocol: matched vocab, BPC not PPL, Morfessor + Indic baselines, morpheme-boundary F1, ParadigmEmb ablation as headline. | Partial — intrinsic metrics only |

**First milestone that matters: an FST coverage number on a real corpus.** An FST that
analyzes 60% of tokens leaves 40% on the loose existential path, and no amount of
probabilistic ranking rescues that. Measure coverage before optimizing anything else.

---

## Before you publish

1. Retrain every baseline at a matched vocabulary budget on the same corpus. The
   current comparison is off-the-shelf and cannot support a fertility claim.
2. Report bits-per-character from a trained LM. Not perplexity — token inventories
   differ, so lower PPL can be pure segmentation artifact.
3. Run the ParadigmEmb ablation (`off` / `random-init` / `P-seeded`). Without it the
   central claim — that a paradigm-seeded embedding learns to unify आयो ↔ आउँछु — is
   unsupported.
4. Either fix the FST-to-tokenizer pipeline, or reframe the contribution as a
   frequency-BPE resource with strong Devanagari coverage. Both are defensible. Claiming
   morphological segmentation while `Morph.Share = 0.000` is not.

---

## License

MIT.

## Citation

```bibtex
@software{himalayantok_nepali,
  title  = {HimalayanTOK: An Akshara-Aware Constrained BPE Tokenizer for Nepali},
  year   = {2026},
  url    = {https://github.com/<you>/HimalayanTOK-Nepali}
}
```