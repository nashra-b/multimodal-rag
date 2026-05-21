#!/usr/bin/env python3
"""
scripts/evaluate.py
--------------------
Retrieval and generation quality evaluation using Ragas 0.4.x

Metrics:
    faithfulness        - Is the answer grounded in the retrieved context?
    answer_relevancy    - Does the answer address the question?
    context_precision   - Are retrieved chunks relevant to the question?
    context_recall      - Do retrieved chunks cover the ground truth answer?

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --eval-file data/eval/my_qa.json
    python scripts/evaluate.py --sample 5 --output results/run1.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

# ── Ragas 0.4.x imports ───────────────────────────────────────────────────────
try:
    from ragas                        import evaluate
    from ragas.metrics                import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.dataset_schema         import SingleTurnSample, EvaluationDataset
    from ragas.llms                   import LangchainLLMWrapper
    from ragas.embeddings             import LangchainEmbeddingsWrapper
    from langchain_openai             import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("        pip install ragas==0.4.3 langchain-openai")
    sys.exit(1)

from src.pipeline    import RAGChain
from src.vectorstore import PineconeClient, HybridRetriever

logger = logging.getLogger("evaluate")

# ── Sample Q&A dataset ────────────────────────────────────────────────────────
SAMPLE_QA = [
    {"question": "What was the total revenue reported?",
     "ground_truth": "Total revenue figures are in the financial summary section."},
    {"question": "What are the key risk factors mentioned?",
     "ground_truth": "Key risk factors are outlined in the risk management section."},
    {"question": "What is the net interest income?",
     "ground_truth": "Net interest income is reported in the financial statements."},
    {"question": "How many employees does the company have?",
     "ground_truth": "Employee headcount is in the human resources section."},
    {"question": "What is the capital adequacy ratio?",
     "ground_truth": "The capital adequacy ratio is in the regulatory capital section."},
    {"question": "Describe the loan portfolio composition.",
     "ground_truth": "Loan portfolio breakdown by category is in the credit section."},
    {"question": "What dividends were declared?",
     "ground_truth": "Dividend information is in the shareholder returns section."},
    {"question": "What technology investments were made?",
     "ground_truth": "Technology investments are described in the strategy section."},
    {"question": "What were operating expenses?",
     "ground_truth": "Operating expenses are broken down in the income statement."},
    {"question": "Summarise the CEO's message.",
     "ground_truth": "The CEO letter appears at the beginning of the annual report."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas 0.4.x evaluation")
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def validate_env() -> None:
    missing = [k for k in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
               if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        sys.exit(1)


def load_qa_pairs(eval_file, sample) -> list[dict]:
    if eval_file:
        with open(eval_file) as f:
            pairs = json.load(f)
    else:
        pairs = SAMPLE_QA
    return pairs[:sample] if sample else pairs


def run_rag_inference(qa_pairs, rag_chain) -> tuple[list[str], list[list[str]]]:
    answers, contexts = [], []
    for i, pair in enumerate(qa_pairs, 1):
        logger.info(f"  [{i}/{len(qa_pairs)}] {pair['question'][:60]}…")
        try:
            result = rag_chain.invoke(pair["question"])
            answers.append(result.get("answer", ""))
            contexts.append([d.page_content for d in result.get("source_documents", [])])
        except Exception as e:
            logger.error(f"  Inference failed: {e}")
            answers.append("")
            contexts.append([""])
    return answers, contexts


def run_ragas_evaluation(qa_pairs, answers, contexts) -> dict:
    """Ragas 0.4.x API — uses SingleTurnSample + EvaluationDataset."""

    # ── Wrap LLM and embeddings for Ragas ─────────────────────────────────────
    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-large")
    )

    # ── Build metric instances (0.4.x requires explicit llm injection) ─────────
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # ── Build dataset ──────────────────────────────────────────────────────────
    samples = [
        SingleTurnSample(
            user_input        = pair["question"],
            response          = answer,
            retrieved_contexts= ctx,
            reference         = pair["ground_truth"],
        )
        for pair, answer, ctx in zip(qa_pairs, answers, contexts)
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info("Running Ragas evaluation …")
    result = evaluate(dataset=dataset, metrics=metrics)
    return result


def print_results(result, output_path) -> None:
    # Ragas 0.4.x result is a dict-like object
    scores = {}
    for metric_name in ["faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"]:
        val = getattr(result, metric_name, None)
        if val is None and hasattr(result, "__getitem__"):
            try:
                val = result[metric_name]
            except Exception:
                val = 0.0
        scores[metric_name] = round(float(val or 0), 4)

    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<25} {'Score':>8}  {'Rating'}")
    print("-" * 55)
    for metric, score in scores.items():
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        grade = ("🟢 Excellent" if score >= 0.90 else
                 "🟡 Good"      if score >= 0.75 else
                 "🟠 Fair"      if score >= 0.60 else "🔴 Needs work")
        print(f"  {metric:<25} {score:>8.4f}  {grade}")
    avg = sum(scores.values()) / len(scores)
    print("-" * 55)
    print(f"  {'Average':<25} {avg:>8.4f}")
    print("=" * 55)
    print(f"\n  Results saved → {output_path}\n")
    return scores


def save_results(scores, qa_pairs, answers, contexts, output) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluated_at":  datetime.now().isoformat(),
        "num_questions": len(qa_pairs),
        "scores":        scores,
        "details": [
            {"question": qa["question"], "ground_truth": qa["ground_truth"],
             "answer": ans, "contexts": ctx}
            for qa, ans, ctx in zip(qa_pairs, answers, contexts)
        ],
    }
    with open(output, "w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    validate_env()

    qa_pairs = load_qa_pairs(args.eval_file, args.sample)

    logger.info("Initialising RAG chain …")
    client   = PineconeClient()
    hybrid   = HybridRetriever(pinecone_index=client.index, all_docs=[])
    chain    = RAGChain(retriever=hybrid.retriever)

    logger.info(f"Running inference on {len(qa_pairs)} question(s) …")
    answers, contexts = run_rag_inference(qa_pairs, chain)

    result = run_ragas_evaluation(qa_pairs, answers, contexts)
    scores = print_results(result, args.output)
    save_results(scores, qa_pairs, answers, contexts, args.output)


if __name__ == "__main__":
    main()