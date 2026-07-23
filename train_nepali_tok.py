#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import HimalayanTokenization
import time
import os
import subprocess
import sys

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Full corpus path (17.4M lines)
CORPUS_PATH = "dataset/book_bank_wiki_corpus_ne.txt"
# Where to store the half-corpus file
HALF_CORPUS_PATH = "dataset/book_bank_wiki_corpus_ne_half.txt"
# Where to store a sample for akshara harvesting (first 1M lines)
AKSHARA_SAMPLE_PATH = "dataset/akshara_sample.txt"
# Checkpoint directory (create if missing)
CHECKPOINT_DIR = "checkpoints"
# Output vocab file
OUTPUT_VOCAB_TSV = "trained_vocab.tsv"

# Training hyperparameters
VOCAB_BUDGET = 100_000           # final vocabulary size
THETA = 100                      # frequency threshold for ambiguous seeds
MIN_WORD_FREQ = 2                # drop word types seen fewer times
PROGRESS_LINES = 500_000         # print build progress every N lines
PROGRESS_MERGES = 1000           # print train progress every N merges
CHECKPOINT_EVERY = 50_000        # save a checkpoint every N merges
CHECKPOINT_KEEP = 3              # keep only last 3 checkpoints

# ============================================================================
# 2. PREPARE DATA (half corpus + akshara sample)
# ============================================================================

def prepare_files():
    """Create half‑corpus and akshara sample if they don't exist."""
    if not os.path.exists(HALF_CORPUS_PATH):
        print("[1/4] Creating half‑corpus file...")
        # Get total lines
        total = int(subprocess.check_output(f"wc -l < {CORPUS_PATH}", shell=True).strip())
        half = total // 2
        print(f"    Total lines: {total:,}, half: {half:,}")
        cmd = f"head -n {half} {CORPUS_PATH} > {HALF_CORPUS_PATH}"
        subprocess.check_call(cmd, shell=True)
        print(f"    Created {HALF_CORPUS_PATH}")
    else:
        print("[1/4] Half‑corpus file already exists, skipping.")

    if not os.path.exists(AKSHARA_SAMPLE_PATH):
        print("[1/4] Creating akshara sample (first 1M lines)...")
        cmd = f"head -n 1000000 {CORPUS_PATH} > {AKSHARA_SAMPLE_PATH}"
        subprocess.check_call(cmd, shell=True)
        print(f"    Created {AKSHARA_SAMPLE_PATH}")
    else:
        print("[1/4] Akshara sample already exists, skipping.")

# ============================================================================
# 3. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    start_total = time.time()

    # Prepare data files
    prepare_files()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --------------------------------------------------------------------
    # 3.1 Initialise tokenizer (mode=LM for aggressive folding)
    # --------------------------------------------------------------------
    print("[2/4] Initialising tokenizer with mode=LM...")
    tok = HimalayanTokenization.PyHimalayanTokenization(mode="LM")

    # --------------------------------------------------------------------
    # 3.2 Harvest aksharas from the sample
    # --------------------------------------------------------------------
    print("[2/4] Harvesting aksharas from sample...")
    start_harvest = time.time()
    aksharas = tok.harvest_aksharas(AKSHARA_SAMPLE_PATH, min_freq=5)
    harvest_time = time.time() - start_harvest
    print(f"    Harvested {len(aksharas)} aksharas in {harvest_time:.2f}s")

    # --------------------------------------------------------------------
    # 3.3 Initialise vocabulary with those aksharas
    # --------------------------------------------------------------------
    print("[3/4] Initialising vocabulary...")
    start_init = time.time()
    tok.initialize_vocab(
        aksharas=aksharas,
        seed_morphemes=[],                 # add common stems manually if you have them
        punctuation=[".", ",", "।", "॥", "?", "!", ";", ":"],
        v_strict=[],                       # never split these (e.g., proper nouns)
        v_ambiguous=[]                     # frequency‑gated seeds (none for now)
    )
    init_time = time.time() - start_init
    print(f"    Initial vocab size: {tok.vocab_size()} (in {init_time:.2f}s)")

    # --------------------------------------------------------------------
    # 3.4 Train on the half‑corpus with checkpoints
    # --------------------------------------------------------------------
    print(f"[4/4] Training on half corpus: {HALF_CORPUS_PATH}")
    print(f"    Target vocab: {VOCAB_BUDGET:,}")
    print(f"    Checkpoints every {CHECKPOINT_EVERY:,} merges → {CHECKPOINT_DIR}")
    start_train = time.time()

    final_vocab_size = tok.train_from_file(
        path=HALF_CORPUS_PATH,
        vocab_budget=VOCAB_BUDGET,
        theta=THETA,
        min_word_freq=MIN_WORD_FREQ,
        progress_lines=PROGRESS_LINES,
        progress_merges=PROGRESS_MERGES,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_keep=CHECKPOINT_KEEP
    )

    train_time = time.time() - start_train
    print(f"    Training completed in {train_time/60:.2f} minutes.")
    print(f"    Final vocab size: {final_vocab_size:,}")

    # --------------------------------------------------------------------
    # 3.5 Save final vocabulary
    # --------------------------------------------------------------------
    print("[5/4] Saving final vocabulary...")
    tok.save_vocab_tsv(OUTPUT_VOCAB_TSV)
    print(f"    Saved to {OUTPUT_VOCAB_TSV}")

    # --------------------------------------------------------------------
    # 3.6 Quick sanity test on a few sentences
    # --------------------------------------------------------------------
    print("\n[6/4] Sample tokenization (first few sentences):")
    test_sentences = [
        "नेपाल एक सुन्दर देश हो।",
        "हामीले विद्यालयमा पढ्यौं।",
        "संविधान सभाबाट पारित भयो।",
    ]
    for sent in test_sentences:
        tokens = tok.tokenize_to_strings(sent)
        print(f"  {sent} -> {tokens}")

    total_time = time.time() - start_total
    print(f"\n✅ ALL DONE in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()