"""
MAIN EXPERIMENTAL ENGINE

Runs the logical calibration experiment:
  - Loads puzzles from dataset.py
  - Computes ground truth via Z3 (solvers.py)
  - Calls GPT-4o-mini with two prompting strategies
  - Extracts logprobs and maps tokens
  - Saves results to CSV and generates charts
"""

import os
import json
import time
import math
import logging
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from dataset import ALL_PUZZLES


# 1. CONFIGURATION

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL       = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS  = 1
TOP_LOGPROBS = 5

VALID_LABELS = {"True", "False", "Unknown", "Paradox"}
DEFAULT_VALID_OUTPUTS = ["True", "False", "Unknown", "Paradox"]

OUTPUT_CSV   = "results.csv"
OUTPUT_CHART = "results_chart.png"


# 2. PROMPT TEMPLATES  (Independent Variables)

STRATEGIES = {
    "Natural": (
        "Read the following story.\n"
        "Reply with a SINGLE word among: True, False, Unknown, Paradox.\n\n"
        "Story:\n{story}\n\n"
        "Answer:"
    ),
    "Formal Persona": (
        "You are a formal verification engine. Internally translate every statement "
        "into propositional axioms and derive the conclusion rigorously.\n"
        "Reply with a SINGLE word among: True, False, Unknown, Paradox.\n"
        "- True: the conclusion is entailed by the premises.\n"
        "- False: the negation of the conclusion is entailed by the premises.\n"
        "- Unknown: the conclusion is contingent.\n"
        "- Paradox: the knowledge base is inconsistent (UNSAT).\n\n"
        "Story:\n{story}\n\n"
        "Answer:"
    ),
}

# 3. PREFIX MAPPING

PREFIX_MAP = {
    "tr":  "True",
    "fa":  "False",
    "un":  "Unknown",
    "pa":  "Paradox",
}


def map_token(raw: str) -> str | None:
    """
    Maps a raw token to the correct label via prefix matching.
    Case-insensitive. Returns None if no prefix matches.

    Ex: "Tr" → "True", "true" → "True", "xyz" → None
    """
    cleaned = raw.strip().lower()
    for prefix, label in PREFIX_MAP.items():
        if cleaned.startswith(prefix):
            return label
    return None


# 4. API CALL

def call_llm(prompt: str) -> dict | None:
    """
    Calls GPT-4o-mini and returns a dictionary with:
        - raw_token      : the generated token (string)
        - confidence     : token probability (0-100%)
        - top_logprobs   : dict {token: probability%} for top 5 alternatives
        - latency_ms     : response time in milliseconds

    Returns None on error, logging a warning.
    """
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
        )
        latency_ms = round((time.time() - start) * 1000, 2)

        content_logprob = response.choices[0].logprobs.content[0]

        raw_token   = content_logprob.token
        confidence  = round(math.exp(content_logprob.logprob) * 100, 2)

        top_lp = {
            item.token: round(math.exp(item.logprob) * 100, 2)
            for item in content_logprob.top_logprobs
        }

        return {
            "raw_token":     raw_token,
            "confidence":    confidence,
            "top_logprobs":  top_lp,
            "latency_ms":    latency_ms,
        }

    except Exception as e:
        logger.warning(f"API error: {e}")
        return None


# 5. MAIN LOOP

def run_experiment() -> list[dict]:
    """
    Runs the full experiment.
    For each puzzle, for each strategy:
        1. Compute ground truth via Z3
        2. Build the prompt
        3. Call the API
        4. Map the token and compare with ground truth
        5. Record the result
    """
    results = []
    errors  = 0

    total_calls = len(ALL_PUZZLES) * len(STRATEGIES)
    logger.info(f"Experiment start — {len(ALL_PUZZLES)} puzzles × {len(STRATEGIES)} strategies = {total_calls} API calls")
    logger.info(f"Model: {MODEL} | temperature={TEMPERATURE} | max_tokens={MAX_TOKENS}")
    logger.info("-" * 60)

    for puzzle in ALL_PUZZLES:
        puzzle_id   = puzzle["id"]
        category    = puzzle["category"]
        story_text  = puzzle["text"]
        z3_func     = puzzle["z3_func"]
        valid_out   = puzzle.get("valid_outputs", DEFAULT_VALID_OUTPUTS)

        # Ground truth (Z3)
        ground_truth = z3_func()
        logger.info(f"[{puzzle_id}] Ground Truth (Z3): {ground_truth}")

        for strategy_name, template in STRATEGIES.items():
            prompt = template.format(story=story_text)

            # API call
            api_result = call_llm(prompt)

            if api_result is None:
                # API error: record with None fields
                errors += 1
                logger.warning(f"  [{strategy_name}] ERROR — execution skipped")
                results.append({
                    "id":            puzzle_id,
                    "category":      category,
                    "strategy":      strategy_name,
                    "raw_token":     None,
                    "mapped_token":  None,
                    "ground_truth":  ground_truth,
                    "correct":       None,
                    "confidence":    None,
                    "top_logprobs":  None,
                    "latency_ms":    None,
                })
                continue

            # Mapping & Validation 
            raw_token     = api_result["raw_token"]
            mapped_token  = map_token(raw_token)

            # Check that the mapped token is within the puzzle's valid output space
            if mapped_token not in valid_out:
                mapped_token = None  # out of valid space → Invalid

            # Comparison with ground truth 
            if mapped_token is None:
                correct = None   # Invalid: excluded from metrics
                label   = "Invalid"
            else:
                correct = int(mapped_token == ground_truth)
                label   = "✓" if correct else "✗"

            logger.info(
                f"  [{strategy_name}] raw='{raw_token}' → mapped='{mapped_token}' | "
                f"GT='{ground_truth}' | {label} | confidence={api_result['confidence']}%"
            )

            results.append({
                "id":            puzzle_id,
                "category":      category,
                "strategy":      strategy_name,
                "raw_token":     raw_token,
                "mapped_token":  mapped_token if mapped_token else "Invalid",
                "ground_truth":  ground_truth,
                "correct":       correct,
                "confidence":    api_result["confidence"],
                "top_logprobs":  json.dumps(api_result["top_logprobs"]),
                "latency_ms":    api_result["latency_ms"],
            })

        logger.info("")  # blank line between puzzles

    # Summary 
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total calls:      {total_calls}")
    logger.info(f"API errors:       {errors}")

    valid_results = [r for r in results if r["correct"] is not None]
    invalid_count = len([r for r in results if r["mapped_token"] == "Invalid"])
    logger.info(f"Invalid tokens:   {invalid_count}")
    logger.info(f"Valid results:    {len(valid_results)}")

    if valid_results:
        accuracy = sum(r["correct"] for r in valid_results) / len(valid_results) * 100
        logger.info(f"Global accuracy:  {accuracy:.1f}%")

    logger.info("=" * 60)

    return results


# 6. CSV EXPORT

CSV_FIELDS = [
    "id", "category", "strategy", "raw_token", "mapped_token",
    "ground_truth", "correct", "confidence", "top_logprobs", "latency_ms",
]


def save_csv(results: list[dict]):
    """Saves results to results.csv."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to '{OUTPUT_CSV}'")


# 7. CHART GENERATION

def generate_charts(results: list[dict]):
    """
    Generates a PNG with 2 side-by-side subplots:
        - Left:  Accuracy by strategy (valid results only)
        - Right: Average Confidence by strategy
    """
    strategies = list(STRATEGIES.keys())

    # Metrics calculation by strategy 
    accuracy_map  = {}
    confidence_map = {}

    for strat in strategies:
        strat_results = [r for r in results if r["strategy"] == strat]

        # Accuracy: valid results only (correct != None)
        valid = [r for r in strat_results if r["correct"] is not None]
        if valid:
            accuracy_map[strat] = sum(r["correct"] for r in valid) / len(valid) * 100
        else:
            accuracy_map[strat] = 0.0

        # Confidence: all results with confidence != None
        with_conf = [r for r in strat_results if r["confidence"] is not None]
        if with_conf:
            confidence_map[strat] = sum(r["confidence"] for r in with_conf) / len(with_conf)
        else:
            confidence_map[strat] = 0.0

    # Colors 
    colors = ["#4A90D9", "#E8734A"]  # Blue, Orange

    # Layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LLM Logical Calibration — GPT-4o-mini", fontsize=14, fontweight="bold", y=1.02)

    x     = np.arange(len(strategies))
    width = 0.5

    # Left subplot: Accuracy 
    ax1 = axes[0]
    bars1 = ax1.bar(x, [accuracy_map[s] for s in strategies], width, color=colors)
    ax1.set_title("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.set_ylabel("%")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # Value labels above bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Right subplot: Confidence
    ax2 = axes[1]
    bars2 = ax2.bar(x, [confidence_map[s] for s in strategies], width, color=colors)
    ax2.set_title("Average Confidence (%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax2.set_ylabel("%")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # Value labels above bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved to '{OUTPUT_CHART}'")


# 8. ENTRY POINT

if __name__ == "__main__":
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START")
    results = run_experiment()
    save_csv(results)
    generate_charts(results)
    logger.info("DONE")