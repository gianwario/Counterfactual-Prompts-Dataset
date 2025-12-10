#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple descriptive statistics + pair count check for the counterfactual
prompts dataset, and export of a JSON summary.

NO command-line arguments: edit the CONFIG section below.

Expectations:
    - For each (bias_type, intent, topic) combination, you *intend* to have
      20 pairs, i.e., 40 rows (since each pair has 2 prompts).

Input CSV columns:
    - topic
    - intent
    - group
    - sentence
    - bias_type
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd


# -------------------- CONFIG -------------------- #
INPUT_CSV = "../data/1_raw_dataset.csv"         # Path to the raw CSV
CSV_SEPARATOR = ","               
SUMMARY_JSON = "../data/stats_summary.json"

EXPECTED_PAIRS = 20               # per (bias_type, intent, topic)
PAIR_SIZE = 2                     # 2 prompts per pair => 40 rows expected
# ------------------------------------------------ #


REQUIRED_COLS = {"topic", "intent", "group", "sentence", "bias_type"}


def load_dataset(path: str, sep: str = ",") -> pd.DataFrame:
    """Load the dataset and check required columns."""
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def compute_simple_stats(df: pd.DataFrame) -> dict:
    """Compute simple descriptive statistics and return them as a dict."""
    stats = {}

    stats["total_rows"] = int(len(df))
    stats["n_unique_topics"] = int(df["topic"].nunique())
    stats["n_unique_intents"] = int(df["intent"].nunique())
    stats["n_unique_groups"] = int(df["group"].nunique())
    stats["n_unique_bias_types"] = int(df["bias_type"].nunique())

    # Value counts
    stats["counts_per_bias_type"] = df["bias_type"].value_counts().to_dict()
    # convert keys to str to be safe for JSON
    stats["counts_per_bias_type"] = {
        str(k): int(v) for k, v in stats["counts_per_bias_type"].items()
    }

    stats["counts_per_intent"] = df["intent"].value_counts().to_dict()
    stats["counts_per_intent"] = {
        str(k): int(v) for k, v in stats["counts_per_intent"].items()
    }


    # bias_type × intent crosstab
    bias_intent = pd.crosstab(df["bias_type"], df["intent"])
    stats["bias_type_by_intent"] = {
        str(idx): {str(col): int(bias_intent.loc[idx, col]) for col in bias_intent.columns}
        for idx in bias_intent.index
    }

    # number of distinct groups per bias_type
    groups_per_bias = (
        df.groupby("bias_type")["group"].nunique().sort_values(ascending=False)
    )
    stats["n_groups_per_bias_type"] = {
        str(k): int(v) for k, v in groups_per_bias.to_dict().items()
    }

    return stats


def compute_pair_count_check(df: pd.DataFrame) -> dict:
    """
    For each (bias_type, intent, topic) combo, check that:

        #pairs = rows / PAIR_SIZE  (ideally == EXPECTED_PAIRS)

    Returns a dict with:
        - distribution of rows per combo
        - counts of perfect / too few / too many / odd-row combos
        - examples of problematic combos (ids only, not full text)
    """
    summary = {}

    key_cols = ["bias_type", "intent", "topic"]

    grouped = (
        df.groupby(key_cols)
        .agg(
            n_rows=("sentence", "size"),
            n_groups=("group", "nunique"),
        )
        .reset_index()
    )

    grouped["n_pairs_float"] = grouped["n_rows"] / PAIR_SIZE

    total_keys = int(len(grouped))
    summary["total_combinations"] = total_keys
    summary["expected_pairs_per_combination"] = EXPECTED_PAIRS
    summary["expected_rows_per_combination"] = EXPECTED_PAIRS * PAIR_SIZE

    # Distribution of n_rows
    n_rows_dist = Counter(grouped["n_rows"])
    summary["row_count_distribution"] = {
        int(k): int(v) for k, v in n_rows_dist.items()
    }

    expected_rows = EXPECTED_PAIRS * PAIR_SIZE

    # odd row count (not divisible by PAIR_SIZE)
    odd_rows = grouped[grouped["n_rows"] % PAIR_SIZE != 0]

    # correct divisibility but not expected size
    divisible = grouped[grouped["n_rows"] % PAIR_SIZE == 0]
    perfect = divisible[divisible["n_rows"] == expected_rows]

    summary["n_perfect_combinations"] = int(len(perfect))
    summary["n_odd_row_combinations"] = int(len(odd_rows))


    return summary


def main():
    print(f"Loading CSV from: {INPUT_CSV}")
    df = load_dataset(INPUT_CSV, sep=CSV_SEPARATOR)

    # ---- Compute simple stats ----
    simple_stats = compute_simple_stats(df)

    # Print a bit to console (optional, minimal)
    print("=" * 80)
    print("BASIC STATS")
    print("=" * 80)
    print(f"Total rows:              {simple_stats['total_rows']}")
    print(f"Unique topics:           {simple_stats['n_unique_topics']}")
    print(f"Unique intents:          {simple_stats['n_unique_intents']}")
    print(f"Unique groups:           {simple_stats['n_unique_groups']}")
    print(f"Unique bias types:       {simple_stats['n_unique_bias_types']}")
    print()

    # ---- Pair count check ----
    pair_check = compute_pair_count_check(df)

    print("=" * 80)
    print("PAIR COUNT CHECK SUMMARY")
    print("=" * 80)
    print(f"Total combinations:      {pair_check['total_combinations']}")
    print(f"Perfect combinations:    {pair_check['n_perfect_combinations']}")
    print(f"Odd-row combinations:    {pair_check['n_odd_row_combinations']}")
    print()

    # ---- Combine and save as JSON ----
    summary = {
        "simple_stats": simple_stats
    }

    out_path = Path(SUMMARY_JSON)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✅ Summary JSON saved to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
