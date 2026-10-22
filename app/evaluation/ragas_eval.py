"""
app/evaluation/ragas_eval.py
-----------------------------
RAGAS (Retrieval Augmented Generation Assessment) evaluation pipeline.

Design Decision:
  WHY RAGAS?
  - RAGAS is the industry-standard automated evaluation framework for RAG systems.
  - It measures 4 key dimensions WITHOUT requiring human-labelled answers:
      1. Faithfulness: Are claims in the answer grounded in retrieved context?
      2. Answer Relevancy: Does the answer address the question asked?
      3. Context Precision: Are the retrieved chunks truly relevant to the query?
      4. Context Recall: Did retrieval capture all necessary information?

  WHY these 4 metrics?
  - Faithfulness prevents hallucination.
  - Answer Relevancy prevents off-topic responses.
  - Context Precision = retrieval quality (wrong chunks → wrong answers).
  - Context Recall = retrieval completeness (missing chunks → incomplete answers).

  EVALUATION FREQUENCY:
  - Run on every code change (CI/CD gate).
  - Track metric drift over time (baseline: faithfulness > 0.85).
  - Trigger alerts if faithfulness drops below 0.70 in production.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_ragas_evaluation(
    questions: List[str],
    ground_truths: List[List[str]],
    answers: List[str],
    contexts: List[List[str]],
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on a set of QA samples.

    Args:
        questions: List of user questions.
        ground_truths: List of reference answer lists (one list per question).
        answers: List of generated answers.
        contexts: List of retrieved context lists (one list per question).

    Returns:
        Dictionary of metric scores and per-sample breakdowns.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.error(
            "RAGAS dependencies not installed: %s. "
            "Run: pip install ragas datasets",
            e,
        )
        return {"error": str(e), "metrics": {}}

    if not (len(questions) == len(ground_truths) == len(answers) == len(contexts)):
        raise ValueError(
            "questions, ground_truths, answers, and contexts must all have the same length."
        )

    # Build HuggingFace Dataset (RAGAS input format)
    data = {
        "question": questions,
        "ground_truth": [" ".join(gt) for gt in ground_truths],
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS evaluation on %d samples...", len(questions))

    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        scores = result.to_pandas().to_dict(orient="list")

        # Aggregate means
        aggregated = {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall": round(float(result["context_recall"]), 4),
            "sample_count": len(questions),
        }

        # Interpretation thresholds (financial domain standards)
        thresholds = {
            "faithfulness": 0.85,       # Critical: no hallucinations
            "answer_relevancy": 0.80,   # Important: on-topic answers
            "context_precision": 0.75,  # Retrieval quality
            "context_recall": 0.70,     # Retrieval completeness
        }

        status = {}
        for metric, threshold in thresholds.items():
            score = aggregated.get(metric, 0)
            status[metric] = "PASS" if score >= threshold else "FAIL"

        logger.info("RAGAS evaluation complete. Scores: %s", aggregated)
        logger.info("Threshold status: %s", status)

        return {
            "metrics": aggregated,
            "threshold_status": status,
            "per_sample": scores,
            "overall_pass": all(v == "PASS" for v in status.values()),
        }

    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"error": str(exc), "metrics": {}}


def build_eval_dataset_from_docs(
    qa_pairs: List[Dict[str, str]],
    retriever,
) -> Dict[str, List]:
    """
    Build RAGAS-format evaluation data from Q&A pairs using the live retriever.

    Args:
        qa_pairs: List of {"question": str, "ground_truth": str} dicts.
        retriever: HybridRetriever instance.

    Returns:
        Dict with questions, ground_truths, answers, and contexts.
    """
    from app.agents.graph import get_graph

    graph = get_graph()
    questions, ground_truths, answers, contexts = [], [], [], []

    for pair in qa_pairs:
        q = pair["question"]
        gt = pair.get("ground_truth", "")

        # Run full pipeline to get actual answer and retrieved docs
        try:
            state = graph.invoke({"query": q, "session_id": "eval"})
            answer = state.get("final_response", "")
            docs = state.get("retrieved_docs", [])
            ctx = [doc.page_content for doc in docs]
        except Exception as exc:
            logger.error("Graph invocation failed for eval question: %s", exc)
            answer = ""
            ctx = []

        questions.append(q)
        ground_truths.append([gt])
        answers.append(answer)
        contexts.append(ctx)

    return {
        "questions": questions,
        "ground_truths": ground_truths,
        "answers": answers,
        "contexts": contexts,
    }
