import pytest

from arbee.tools.bayesian import bayesian_calculate_tool


@pytest.mark.asyncio
async def test_bayesian_tool_converts_estimated_llr_field() -> None:
    evidence_items = [
        {
            "id": "ev1",
            "estimated_LLR": 0.8,
            "support": "pro",
            "verifiability_score": 0.9,
            "independence_score": 0.8,
            "recency_score": 1.0,
        }
    ]

    result = await bayesian_calculate_tool(prior_p=0.5, evidence_items=evidence_items)

    assert result["total_adjusted_LLR"] > 0
    assert result["evidence_summary"][0]["LLR"] == pytest.approx(0.8, rel=1e-9)


@pytest.mark.asyncio
async def test_bayesian_tool_aligns_llr_with_support_direction() -> None:
    evidence_items = [
        {
            "id": "ev_con",
            "estimated_LLR": 0.6,
            "support": "con",
            "verifiability_score": 1.0,
            "independence_score": 1.0,
            "recency_score": 1.0,
        }
    ]

    result = await bayesian_calculate_tool(prior_p=0.5, evidence_items=evidence_items)

    assert result["evidence_summary"][0]["LLR"] == pytest.approx(-0.6, rel=1e-9)
    assert result["total_adjusted_LLR"] < 0
