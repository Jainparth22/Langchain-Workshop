"""
app/api/routes/evaluate.py
---------------------------
POST /evaluate — Run RAGAS evaluation on the current knowledge base.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.evaluation.ragas_eval import run_ragas_evaluation

router = APIRouter()
logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[str]


class EvalRequest(BaseModel):
    samples: List[EvalSample]


@router.post(
    "",
    summary="Run RAGAS quality evaluation",
    description=(
        "Evaluate the RAG pipeline using RAGAS metrics: "
        "faithfulness, answer_relevancy, context_precision, context_recall. "
        "Pass pre-collected samples (question + generated_answer + retrieved_contexts + ground_truth)."
    ),
)
async def evaluate(request: EvalRequest) -> dict:
    """Execute RAGAS evaluation and return metric scores."""
    if not request.samples:
        raise HTTPException(status_code=400, detail="At least one sample is required.")

    questions = [s.question for s in request.samples]
    ground_truths = [[s.ground_truth] for s in request.samples]
    answers = [s.generated_answer for s in request.samples]
    contexts = [s.retrieved_contexts for s in request.samples]

    result = run_ragas_evaluation(questions, ground_truths, answers, contexts)

    if "error" in result:
        raise HTTPException(
            status_code=503,
            detail=f"RAGAS evaluation failed: {result['error']}",
        )

    return result
