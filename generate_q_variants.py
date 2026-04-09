"""
(Optional) Generate question variants using Claude API.

Expands each QA pair with 3 alternative phrasings to improve
semantic search recall. Not required -- build_index.py works
fine with manually written questions.

Usage:
    python generate_q_variants.py input.json output_expanded.json

Features:
    - Checkpoints progress every 20 queries (auto-resumes on restart)
    - Keeps original question even if variant generation fails
    - Rate-limit aware with automatic retry

Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import time
import logging
import argparse

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate-variants")

CHECKPOINT_EVERY = 20
MODEL = "claude-sonnet-4-5-20250929"

PROMPT = """You are helping improve a chatbot's semantic search.
Given an original question, generate 3 alternative ways someone might ask
the same question.

Requirements:
- Each variant must be meaningfully DIFFERENT in wording and structure
- All variants must sound like something a real person would naturally say
- Include: one casual/short version, one rephrased version, one keyword-style version
- Do NOT change the intent or broaden the scope

Example:
Original Q: Where is the Grand Ballroom located?
Variants:
How do I get to the Grand Ballroom?
Grand Ballroom location
Which floor is the Grand Ballroom on?

Now generate 3 variants for:
Q: {question}

Output ONLY the 3 questions, one per line, no numbering, no quotes:"""


def generate_variants(client: anthropic.Anthropic, question: str) -> list[str]:
    """Call Claude API to generate 3 question variants."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(question=question)}],
    )
    text = response.content[0].text.strip()
    variants = [line.strip() for line in text.split("\n") if line.strip()]
    return variants[:3]


def load_checkpoint(checkpoint_file: str) -> tuple[list[dict], int]:
    """Load progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        logger.info(
            "Resuming from checkpoint: %d/%d done", ckpt["processed"], ckpt["total"]
        )
        return ckpt["expanded"], ckpt["processed"]
    return [], 0


def save_checkpoint(
    checkpoint_file: str, expanded: list[dict], processed: int, total: int
):
    """Save progress to checkpoint file."""
    with open(checkpoint_file, "w") as f:
        json.dump(
            {"expanded": expanded, "processed": processed, "total": total},
            f,
            ensure_ascii=False,
        )
    logger.info("Checkpoint saved: %d/%d", processed, total)


def main():
    parser = argparse.ArgumentParser(
        description="Generate question variants using Claude API"
    )
    parser.add_argument("input_file", help="Input QA JSON file")
    parser.add_argument("output_file", help="Output expanded JSON file")
    parser.add_argument(
        "--checkpoint",
        default="variant_checkpoint.json",
        help="Checkpoint file path (default: variant_checkpoint.json)",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(args.input_file) as f:
        qa_pairs = json.load(f)

    expanded, start_idx = load_checkpoint(args.checkpoint)

    logger.info("Total QA pairs: %d", len(qa_pairs))
    logger.info("Starting from index: %d", start_idx)

    errors = 0
    for i in range(start_idx, len(qa_pairs)):
        qa = qa_pairs[i]
        original_q = qa.get("q") or qa.get("question") or ""
        original_a = qa.get("a") or qa.get("answer") or ""

        try:
            variants = generate_variants(client, original_q)
            expanded.append({"questions": [original_q] + variants, "a": original_a})
            logger.info("[%d/%d] Q: %s", i + 1, len(qa_pairs), original_q[:60])
            for v in variants:
                logger.info("  variant: %s", v)

        except Exception as e:
            errors += 1
            logger.error("[%d/%d] ERROR: %s", i + 1, len(qa_pairs), e)
            expanded.append({"questions": [original_q], "a": original_a})

            if "rate_limit" in str(e).lower():
                logger.warning("Rate limited. Waiting 30s...")
                time.sleep(30)
                try:
                    variants = generate_variants(client, original_q)
                    expanded[-1]["questions"] += variants
                    logger.info("  Retry succeeded, +%d variants", len(variants))
                    errors -= 1
                except Exception as e2:
                    logger.error("  Retry failed: %s", e2)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(args.checkpoint, expanded, i + 1, len(qa_pairs))

        time.sleep(0.3)

    # Final save
    with open(args.output_file, "w") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)

    total_questions = sum(len(item["questions"]) for item in expanded)
    logger.info("Done. Original: %d pairs, Expanded: %d total questions, Errors: %d",
                len(qa_pairs), total_questions, errors)
    logger.info("Saved to %s", args.output_file)


if __name__ == "__main__":
    main()
