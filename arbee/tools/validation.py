"""
Validation Tools
Help agents validate their work and assumptions
"""
from typing import Dict, Any
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
async def validate_prior_tool(
    prior_p: float,
    market_question: str,
    justification: str
) -> Dict[str, Any]:
    """
    Validate that a prior probability is reasonable.

    Use this to check if a prior is well-calibrated before using it in analysis.

    Args:
        prior_p: Proposed prior probability (0.0 to 1.0)
        market_question: The market question
        justification: Reasoning for this prior

    Returns:
        Dict with is_valid, feedback, and suggested_range

    Example:
        >>> result = await validate_prior_tool(
        ...     0.85,
        ...     "Will it rain tomorrow?",
        ...     "Weather forecast shows 85% chance"
        ... )
    """
    try:
        logger.info(f"✅ Validating prior: p={prior_p:.2%}")

        issues = []

        # Check range
        if not (0.0 <= prior_p <= 1.0):
            issues.append(f"Prior {prior_p} is outside valid range [0, 1]")

        # Check for extreme values (should be rare)
        if prior_p < 0.01 or prior_p > 0.99:
            issues.append(
                f"Prior {prior_p:.1%} is very extreme. "
                "Are you certain? Most events should have some uncertainty."
            )

        # Check for unjustified 50%
        if abs(prior_p - 0.5) < 0.01 and 'uncertain' not in justification.lower():
            issues.append(
                "Prior of exactly 50% suggests maximum uncertainty. "
                "Is there really no information to inform a directional prior?"
            )

        # Check justification length
        if len(justification) < 20:
            issues.append("Justification is very brief. Please provide more reasoning.")

        is_valid = len(issues) == 0

        return {
            'is_valid': is_valid,
            'prior_p': prior_p,
            'issues': issues,
            'feedback': "Prior looks reasonable" if is_valid else "; ".join(issues),
            'suggested_range': (max(0.01, prior_p - 0.1), min(0.99, prior_p + 0.1))
        }

    except Exception as e:
        logger.error(f"❌ Prior validation failed: {e}")
        return {'is_valid': False, 'error': str(e)}


@tool
async def check_llr_calibration_tool(
    llr: float,
    source_type: str,
    verifiability: float
) -> Dict[str, Any]:
    """
    Check if an LLR is properly calibrated for source type and quality.

    Use this to validate evidence LLRs before Bayesian aggregation.

    Args:
        llr: Log-likelihood ratio
        source_type: Source type (primary, high_quality_secondary, secondary, weak)
        verifiability: Verifiability score (0-1)

    Returns:
        Dict with is_calibrated, expected_range, and recommendations
    """
    from arbee.tools.bayesian import validate_llr_calibration_tool
    return await validate_llr_calibration_tool(llr, source_type)


@tool
async def validate_search_results_tool(
    search_results: list,
    min_results: int = 3
) -> Dict[str, Any]:
    """
    Validate that search results are sufficient for analysis.

    Use this to check if you've gathered enough evidence before proceeding.

    Args:
        search_results: List of search results
        min_results: Minimum acceptable number of results

    Returns:
        Dict with is_sufficient, result_count, and recommendations

    Example:
        >>> validation = await validate_search_results_tool(results, min_results=5)
        >>> if not validation['is_sufficient']:
        ...     # Do more searches
    """
    try:
        result_count = len(search_results)

        is_sufficient = result_count >= min_results

        if is_sufficient:
            feedback = f"Found {result_count} results, sufficient for analysis"
        else:
            feedback = f"Only {result_count} results found, recommend {min_results - result_count} more searches"

        # Check diversity
        if result_count > 0:
            domains = set()
            for result in search_results:
                url = result.get('url', '')
                domain = url.split('/')[2] if '/' in url and len(url.split('/')) > 2 else 'unknown'
                domains.add(domain)

            diversity_ratio = len(domains) / result_count if result_count > 0 else 0

            if diversity_ratio < 0.5:
                feedback += f" | Low source diversity ({len(domains)} domains for {result_count} results)"

        return {
            'is_sufficient': is_sufficient,
            'result_count': result_count,
            'min_required': min_results,
            'feedback': feedback
        }

    except Exception as e:
        logger.error(f"❌ Search results validation failed: {e}")
        return {'is_sufficient': False, 'error': str(e)}
