"""Transparent retrieval metrics for the Text-RAG Pipeline class.

This script evaluates retrieval separately from generation, matching the
diagnostic approach in the presentation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from utils import open_collection, retrieve_chunks

BASE_DIR = Path(__file__).parent


def load_gold_questions(path: Path) -> list[dict[str, Any]]:
    """Load questions, expected answers, and relevant source IDs.

    Args:
        path: Location of ``gold_questions.json``.

    Returns:
        Gold evaluation records.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate how clean the first k retrieved results are.

    Args:
        retrieved_ids: Source IDs returned in ranked order.
        relevant_ids: Source IDs known to contain the answer.
        k: Ranking cutoff.

    Returns:
        Precision@k between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = set(relevant_ids)
    hits = sum(source_id in relevant for source_id in retrieved_ids[:k])
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate how much known evidence appears in the first k results.

    Args:
        retrieved_ids: Source IDs returned in ranked order.
        relevant_ids: Source IDs known to contain the answer.
        k: Ranking cutoff.

    Returns:
        Recall@k between 0.0 and 1.0.
    """
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(found) / len(set(relevant_ids))


def answer_f1(prediction: str, expected: str) -> float:
    """Calculate simple token-overlap F1 for short factual answers.

    Args:
        prediction: Generated answer.
        expected: Reference answer.

    Returns:
        Token F1 between 0.0 and 1.0.
    """
    def tokens(text: str) -> list[str]:
        """Lowercase text and keep simple word and number tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    predicted_tokens = tokens(prediction)
    expected_tokens = tokens(expected)
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)

    overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(
    collection: Any, gold_questions: list[dict[str, Any]], top_k: int = 4
) -> list[dict[str, Any]]:
    """Run each gold question and calculate retrieval metrics.

    Args:
        collection: Populated Chroma collection.
        gold_questions: Records returned by :func:`load_gold_questions`.
        top_k: Number of retrieved chunks per question.

    Returns:
        Per-question precision and recall rows.
    """
    rows = []
    for item in gold_questions:
        retrieved = retrieve_chunks(collection, item["question"], top_k)
        retrieved_ids = [row["metadata"]["source_id"] for row in retrieved]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieved_ids": retrieved_ids,
                "precision": precision_at_k(
                    retrieved_ids, item["relevant_source_ids"], top_k
                ),
                "recall": recall_at_k(
                    retrieved_ids, item["relevant_source_ids"], top_k
                ),
            }
        )
    return rows


def main() -> None:
    """Evaluate the persisted collection and print per-question metrics."""
    gold_questions = load_gold_questions(BASE_DIR / "data" / "gold_questions.json")
    client, collection = open_collection(BASE_DIR / "chroma_store")
    if collection.count() == 0:
        client.close()
        raise RuntimeError("Run rag_demo.py before evaluate.py so the corpus is indexed.")

    rows = evaluate_retrieval(collection, gold_questions, top_k=4)
    for row in rows:
        print(
            f"{row['id']}: precision@4={row['precision']:.2f}, "
            f"recall@4={row['recall']:.2f}"
        )
        print(f"   {row['question']}")

    mean_precision = sum(row["precision"] for row in rows) / len(rows)
    mean_recall = sum(row["recall"] for row in rows) / len(rows)
    print(f"\nMean precision@4: {mean_precision:.2f}")
    print(f"Mean recall@4: {mean_recall:.2f}")
    client.close()


if __name__ == "__main__":
    main()
