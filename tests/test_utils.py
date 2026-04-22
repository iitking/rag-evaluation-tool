import json
import pytest

from backend import config
from backend.utils import (
    calculate_cost_estimate,
    export_markdown_report,
    generate_sample_dataset,
    mean_or_none,
    parse_embedders,
    parse_questions,
    parse_strategies,
    retrieval_hit,
)


def test_parse_questions_with_and_without_expected():
    raw = "What is the notice period? || 30 days\n\n  How much leave?  \n"
    items = parse_questions(raw)
    assert items == [
        {"question": "What is the notice period?", "expected": "30 days"},
        {"question": "How much leave?", "expected": None},
    ]


def test_parse_questions_empty():
    assert parse_questions("") == []
    assert parse_questions("   \n || only phrase") == []


def test_parse_strategies_defaults_and_custom():
    assert parse_strategies("") == config.DEFAULT_STRATEGIES
    out = parse_strategies(json.dumps([{"chunk": 300, "overlap": 30}]))
    assert out == [{"name": "chunk300", "chunk": 300, "overlap": 30}]


@pytest.mark.parametrize(
    "bad",
    [
        json.dumps([{"chunk": 50, "overlap": 0}]),      # chunk too small
        json.dumps([{"chunk": 500, "overlap": 500}]),   # overlap not smaller than chunk
        json.dumps([]),                                  # empty list
        json.dumps([{"overlap": 10}]),                   # missing chunk
    ],
)
def test_parse_strategies_rejects_bad_input(bad):
    with pytest.raises((ValueError, KeyError)):
        parse_strategies(bad)


def test_parse_embedders():
    assert parse_embedders("") == [config.ALLOWED_EMBEDDERS[0]]
    name = config.ALLOWED_EMBEDDERS[-1]
    assert parse_embedders(f"{name},{name}") == [name]
    with pytest.raises(ValueError):
        parse_embedders("not-a-real-model")


def test_retrieval_hit():
    assert retrieval_hit("Notice period is 30 DAYS.", "30 days") is True
    assert retrieval_hit("nothing here", "30 days") is False
    assert retrieval_hit("anything", None) is None


def test_retrieval_hit_multiple_alternatives():
    assert retrieval_hit("We offer one month notice.", "30 days; one month") is True
    assert retrieval_hit("We offer two weeks.", "30 days; one month") is False


def test_mean_or_none():
    assert mean_or_none([8, 6, None]) == 7.0
    assert mean_or_none([None, None]) is None


def test_calculate_cost_estimate():
    cost = calculate_cost_estimate(1000, 500)
    assert cost["groq_cost_usd"] == 0.0
    assert cost["total_tokens"] == 1500
    assert cost["estimated_commercial_usd"] > 0


def test_generate_sample_dataset():
    sample = generate_sample_dataset()
    assert "filename" in sample
    assert "content" in sample
    assert len(sample["content"]) > 100
    assert "questions" in sample
    assert len(sample["strategies"]) > 0
    assert len(sample["embedders"]) > 0


def test_export_markdown_report():
    sample_results = [
        {
            "summary": {
                "strategy": "Micro (256)",
                "embedder": "all-MiniLM-L6-v2",
                "chunk_size": 256,
                "overlap": 20,
                "overall": 8.5,
                "avg_relevance": 9.0,
                "avg_faithfulness": 8.0,
                "retrieval_hit_rate": 1.0,
                "avg_latency_s": 0.45,
                "num_chunks": 12,
                "total_input_tokens": 500,
                "total_output_tokens": 120,
            },
            "details": [
                {
                    "question": "What is RAG?",
                    "expected": "Retrieval",
                    "answer": "Retrieval-Augmented Generation.",
                    "relevance_score": 9,
                    "faithfulness_score": 8,
                    "retrieval_hit": True,
                    "latency_s": 0.45,
                    "input_tokens": 500,
                    "output_tokens": 120,
                    "explanation": "Accurate and grounded.",
                }
            ],
        }
    ]
    report = export_markdown_report(sample_results, {"generator_model": "llama-3.3-70b-versatile", "judge_model": "openai/gpt-oss-120b", "top_k": 4})
    assert "# 🧪 RAG Evaluation & Optimization Benchmark Report" in report
    assert "Micro (256)" in report
    assert "all-MiniLM-L6-v2" in report
    assert "8.5" in report
