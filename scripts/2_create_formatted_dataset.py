import json
from pathlib import Path
import pandas as pd


# -------------------- CONFIG -------------------- #
INPUT_CSV = "../data/1_raw_dataset.csv"          
OUTPUT_JSONL = "../data/dataset.jsonl"
CSV_SEPARATOR = ","                
# ------------------------------------------------- #

REQUIRED_COLS = {"topic", "intent", "group", "sentence", "bias_type"}


def load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_pairs(df: pd.DataFrame):
    """
    Construct pairs by assigning pair_index within each group for each
    (bias_type, intent, topic) combination.
    """
    df = df.copy()

    # Step 1: assign pair index
    df["pair_index"] = df.groupby(
        ["bias_type", "intent", "topic", "group"]
    ).cumcount()

    # Step 2: group pairs
    pairs = []
    for (bias_type, intent, topic, pair_idx), sub in df.groupby(
        ["bias_type", "intent", "topic", "pair_index"]
    ):
        sub = sub.sort_values("group")

        pair_id = f"{bias_type}||{intent}||{topic}||{pair_idx}"

        prompts = [
            {"group": row.group, "sentence": row.sentence}
            for row in sub.itertuples()
        ]

        pairs.append(
            {
                "id": pair_id,
                "bias_type": bias_type,
                "intent": intent,
                "topic": topic,
                "pair_index": int(pair_idx),
                "groups": [p["group"] for p in prompts],
                "prompts": prompts,
            }
        )

    return pairs


def save_jsonl(pairs, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print(f"Loading CSV: {INPUT_CSV}")
    df = load_csv(INPUT_CSV, sep=CSV_SEPARATOR)

    print("Building pairs...")
    pairs = build_pairs(df)

    print(f"Saving JSONL: {OUTPUT_JSONL}")
    save_jsonl(pairs, OUTPUT_JSONL)

    print(f"✅ Done. Total pairs created: {len(pairs)}")


if __name__ == "__main__":
    main()
