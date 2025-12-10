import json
import random
import textwrap
from pathlib import Path
import requests
import time

# -------------------- CONFIG -------------------- #
PAIRS_FILE = "../data/dataset.jsonl"   # Path to the dataset

# Filter settings
SELECTED_INTENT = "Question"        
SELECTED_BIAS_TYPE = "race-color"   

# How many pairs (max) to sample from the filtered subset
NUM_SAMPLED_PAIRS = 5

# LLM usage
USE_LLM = True  # Set to True if you implement call_llm()

OUTPUT_JSON = "3_example_usage_1_results.json"


GOOGLE_API_KEY=""
GEMINI_MODEL = "gemini-2.5-flash" 
# Global configuration for free-tier safety
FREE_TIER_MIN_DELAY = 30      # seconds between calls
MAX_RETRIES = 5                # how many times to retry on failure
# ------------------------------------------------- #


# -------------------- Dataset loading -------------------- #
def load_pairs(path: str):
    """Load JSONL dataset into a list of dicts."""
    pairs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs

# -------------------- Google Gemini client -------------------- #
def get_gemini_client():
    """
    Initialize and return the Google AI client, or None if not configured.
    """
    if not USE_LLM:
        return None


    try:
        import google.genai as genai
    except ImportError:
        print("⚠ google-genai package not installed. Run: pip install google-genai")
        return None

    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client

def call_llm(client, prompt: str) -> str:
    """
    Call Gemini with a built-in safe delay + retry logic for free-tier users.

    Features:
      - Ensures a WAIT between calls (FREE_TIER_MIN_DELAY).
      - Retries on transient errors (429, quota, timeouts).
      - Exponential backoff with jitter.
      - Graceful failure after MAX_RETRIES.

    Returns:
      - The model text output, OR a fallback string on error.
    """

    if client is None:
        return "(LLM disabled or not configured.)"

    # Wait between calls — essential for free usage
    time.sleep(FREE_TIER_MIN_DELAY)

    attempt = 0
    last_exception = None

    while attempt < MAX_RETRIES:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            # Extract text (google-genai SDK)
            return getattr(response, "text", str(response))

        except Exception as e:
            last_exception = e
            attempt += 1

            sleep_time = 15
            print(f"⚠️ Gemini error on attempt {attempt}/{MAX_RETRIES}: {e}")
            print(f"   Waiting {sleep_time:.2f}s before retrying…")
            time.sleep(sleep_time)

    return f"(Error querying Gemini after {MAX_RETRIES} retries: {last_exception})"


"""
# -------------------- LLM call (generic template) -------------------- #
def call_llm(prompt: str) -> str:
 
    Generic LLM call.

    This is a TEMPLATE. You can implement it to call:
      - LM Studio (local server)
      - OpenAI-compatible APIs
      - Any custom endpoint

    By default, if USE_LLM is False, it returns a placeholder string.
    If USE_LLM is True, you need to fill in the HTTP request details.

    Example for an OpenAI-compatible HTTP endpoint (LM Studio, OpenAI, etc.):
    -------------------------------------------------------------------------
    
    

    if not USE_LLM:
        return "(LLM disabled: set USE_LLM = True and implement call_llm().)"

    url = "http://localhost:1234/v1/chat/completions"  # or your own endpoint
    headers = {"Content-Type": "application/json", "Authorization": "Bearer YOUR_KEY"}
    payload = {
        "model": "your-model-name",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    data = response.json()
    return data["choices"][0]["message"]["content"]
"""


# -------------------- Simple answer comparison -------------------- #
def simple_compare(a: str, b: str):
    """
    Very naive comparison between two answers (length + Jaccard overlap).
    """
    tokens_a = a.split()
    tokens_b = b.split()
    set_a = set(t.lower() for t in tokens_a)
    set_b = set(t.lower() for t in tokens_b)

    if not set_a and not set_b:
        jaccard = 0.0
    else:
        jaccard = len(set_a & set_b) / len(set_a | set_b)

    return {
        "len_a": len(tokens_a),
        "len_b": len(tokens_b),
        "jaccard_overlap": jaccard,
    }




def main():
    print(f"Loading dataset: {PAIRS_FILE}")
    all_pairs = load_pairs(PAIRS_FILE)
    print(f"Total pairs: {len(all_pairs)}")

    # Filter by selected intent + bias type
    filtered = [
        p for p in all_pairs
        if p["intent"] == SELECTED_INTENT
        and p["bias_type"] == SELECTED_BIAS_TYPE
    ]

    print(f"Filtered pairs ({SELECTED_INTENT}, {SELECTED_BIAS_TYPE}): {len(filtered)}")

    if not filtered:
        print("No matching pairs. Update SELECTED_INTENT / SELECTED_BIAS_TYPE.")
        return

    # Sample N pairs
    n = min(NUM_SAMPLED_PAIRS, len(filtered))
    sampled = random.sample(filtered, n)
    client = get_gemini_client()

    # This will store everything we save
    output_data = {
        "intent": SELECTED_INTENT,
        "bias_type": SELECTED_BIAS_TYPE,
        "num_sampled": n,
        "gemini_model": GEMINI_MODEL if client is not None else None,
        "results": []
    }

    # Iterate and evaluate
    for item in sampled:
        print("=" * 80)
        print(f"Pair ID: {item['id']}")
        print(f"Groups: {item['groups']}")

        pair_record = {
            "pair_id": item["id"],
            "topic": item["topic"],
            "groups": item["groups"],
            "prompts": [],
            "comparisons": []
        }

        answers = []
        prompts = item["prompts"]

        for p in prompts:
            print(f"\n[Group: {p['group']}]")
            print(textwrap.indent(p["sentence"], "  "))

            try:
                ans = call_llm(client, p["sentence"])
            except Exception as e:
                ans = f"(LLM error or disabled: {e})"

            print("\nLLM answer:")
            print(textwrap.indent(ans, "    "))

            pair_record["prompts"].append({
                "group": p["group"],
                "sentence": p["sentence"],
                "answer": ans
            })

            answers.append(ans)

        # Compare answers (if LLM used)
        if USE_LLM and len(answers) >= 2:
            comp = simple_compare(answers[0], answers[1])
            pair_record["comparisons"].append(comp)

            print("\nComparison:")
            for k, v in comp.items():
                print(f"  {k}: {v}")

        output_data["results"].append(pair_record)

    # Save JSON results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()