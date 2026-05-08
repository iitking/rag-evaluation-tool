import pytest
from backend.evaluator import parse_judge_output


def test_parse_judge_output_clean_json():
    raw = '{"relevance_score": 9, "faithfulness_score": 8, "explanation": "Grounded and direct."}'
    parsed = parse_judge_output(raw)
    assert parsed["relevance_score"] == 9
    assert parsed["faithfulness_score"] == 8
    assert parsed["explanation"] == "Grounded and direct."


def test_parse_judge_output_markdown_block():
    raw = """Here is my review:
```json
{
  "relevance_score": 10,
  "faithfulness_score": 9,
  "explanation": "Excellent response."
}
```
"""
    parsed = parse_judge_output(raw)
    assert parsed["relevance_score"] == 10
    assert parsed["faithfulness_score"] == 9
    assert parsed["explanation"] == "Excellent response."


def test_parse_judge_output_with_reasoning_tags():
    raw = """<think>
The user asked for the capital of France. The context says Paris.
The answer is Paris. It is fully faithful and relevant.
</think>
```json
{
  "relevance_score": 10,
  "faithfulness_score": 10,
  "explanation": "Completely faithful."
}
```"""
    parsed = parse_judge_output(raw)
    assert parsed["relevance_score"] == 10
    assert parsed["faithfulness_score"] == 10


def test_parse_judge_output_regex_fallback():
    raw = """relevance_score: 8
faithfulness_score: 7
explanation: "Good but missing a minor detail"
"""
    parsed = parse_judge_output(raw)
    assert parsed["relevance_score"] == 8
    assert parsed["faithfulness_score"] == 7
    assert "Good but missing" in parsed["explanation"]


def test_parse_judge_output_clamping():
    raw = '{"relevance_score": 15, "faithfulness_score": -2, "explanation": "Out of range test."}'
    parsed = parse_judge_output(raw)
    assert parsed["relevance_score"] == 10
    assert parsed["faithfulness_score"] == 1


def test_parse_judge_output_invalid_raises():
    with pytest.raises(ValueError):
        parse_judge_output("completely random nonsense without scores")
