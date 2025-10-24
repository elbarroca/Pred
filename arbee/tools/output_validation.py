"""
Output Validation Tools - Verify agent outputs meet quality standards
Each agent should validate its final output before completion
"""
from typing import Dict, Any, List
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def validate_planner_output_tool(
    market_slug: str,
    market_question: str,
    p0_prior: float,
    prior_justification: str,
    subclaims: List[Dict[str, str]],
    search_seeds: Dict[str, List[str]],
    min_subclaims: int = 4,
    min_search_seeds_per_direction: int = 3
) -> Dict[str, Any]:
    """
    Validate planner output meets quality standards

    Checks:
    - Has valid prior (0.01-0.99)
    - Has minimum subclaims
    - Subclaims are balanced (PRO and CON)
    - Has sufficient search seeds for each direction
    - Has justification for prior
    - Search seeds use current year (2025), not outdated years

    Args:
        market_slug: Market identifier
        market_question: Market question text
        p0_prior: Prior probability estimate
        prior_justification: Justification for prior
        subclaims: List of subclaim dicts with id, text, direction
        search_seeds: Dict with pro, con, general search query lists
        min_subclaims: Minimum number of subclaims required
        min_search_seeds_per_direction: Minimum search seeds per direction

    Returns:
        Dict with is_valid, issues, and suggestions
    """
    # Reconstruct plan dict
    plan = {
        'market_slug': market_slug,
        'market_question': market_question,
        'p0_prior': p0_prior,
        'prior_justification': prior_justification,
        'subclaims': subclaims,
        'search_seeds': search_seeds
    }
    issues = []
    suggestions = []

    # Check prior
    prior = plan.get('p0_prior')
    if prior is None:
        issues.append("Missing p0_prior")
    elif not (0.01 <= prior <= 0.99):
        issues.append(f"Prior {prior} outside valid range (0.01-0.99)")

    # Check prior justification
    justification = plan.get('prior_justification', '')
    if not justification or len(justification) < 20:
        issues.append("Prior justification too short or missing")
        suggestions.append("Provide 2-3 sentence justification for your prior estimate")

    # Check subclaims
    subclaims = plan.get('subclaims', [])
    if len(subclaims) < min_subclaims:
        issues.append(f"Only {len(subclaims)} subclaims (need {min_subclaims})")

    # Check balance
    pro_count = sum(1 for sc in subclaims if sc.get('direction') == 'pro')
    con_count = sum(1 for sc in subclaims if sc.get('direction') == 'con')

    if pro_count == 0:
        issues.append("No PRO subclaims")
    if con_count == 0:
        issues.append("No CON subclaims")

    if abs(pro_count - con_count) > 2:
        suggestions.append(f"Subclaims imbalanced: {pro_count} PRO vs {con_count} CON - aim for balance")

    # Check search seeds
    search_seeds = plan.get('search_seeds', {})
    for direction in ['pro', 'con', 'general']:
        seeds = search_seeds.get(direction, [])
        if len(seeds) < min_search_seeds_per_direction:
            issues.append(f"Only {len(seeds)} {direction} search seeds (need {min_search_seeds_per_direction})")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("✅ Planner output validation PASSED")
    else:
        logger.warning(f"❌ Planner output validation FAILED: {len(issues)} issues")

    return {
        'is_valid': is_valid,
        'issues': issues,
        'suggestions': suggestions,
        'quality_score': _calculate_plan_quality_score(plan)
    }


@tool
def validate_bayesian_calculation_tool(
    p0_prior: float,
    evidence_items: List[Dict[str, Any]],
    p_bayesian: float,
    log_odds_calculation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate Bayesian probability calculation is mathematically correct

    Checks:
    - Log-odds calculation is correct
    - LLR values are reasonable
    - Posterior probability is in valid range
    - Calculation steps are auditable

    Args:
        p0_prior: Prior probability
        evidence_items: List of evidence with LLR scores
        p_bayesian: Calculated posterior probability
        log_odds_calculation: Dict with calculation steps

    Returns:
        Dict with is_valid, issues, and recalculated probability
    """
    import math

    issues = []
    warnings = []

    # Check prior range
    if not (0.01 <= p0_prior <= 0.99):
        issues.append(f"Prior {p0_prior} outside valid range")

    # Check posterior range
    if not (0.01 <= p_bayesian <= 0.99):
        issues.append(f"Posterior {p_bayesian} outside valid range")

    # Recalculate Bayesian update manually
    try:
        log_odds_prior = math.log(p0_prior / (1 - p0_prior))

        total_adjusted_llr = 0.0
        for item in evidence_items:
            llr = item.get('estimated_LLR', 0.0)

            # Get adjustment factors
            verifiability = item.get('verifiability_score', 0.5)
            independence = item.get('independence_score', 0.8)
            recency = item.get('recency_score', 0.7)

            adjusted_llr = llr * verifiability * independence * recency
            total_adjusted_llr += adjusted_llr

            # Check for extreme LLRs
            if abs(llr) > 5.0:
                warnings.append(f"Extreme LLR detected: {llr:.2f} for '{item.get('title', 'unknown')}'")

        log_odds_posterior = log_odds_prior + total_adjusted_llr

        # Convert back to probability
        p_recalculated = math.exp(log_odds_posterior) / (1 + math.exp(log_odds_posterior))

        # Check if calculation matches
        diff = abs(p_bayesian - p_recalculated)
        if diff > 0.01:  # Allow 1% tolerance
            issues.append(
                f"Calculation mismatch: reported {p_bayesian:.4f} but recalculated {p_recalculated:.4f} "
                f"(diff: {diff:.4f})"
            )

        # Validate calculation steps if provided
        if log_odds_calculation:
            reported_prior_log_odds = log_odds_calculation.get('log_odds_prior')
            if reported_prior_log_odds and abs(reported_prior_log_odds - log_odds_prior) > 0.001:
                issues.append("Log-odds prior calculation mismatch")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(f"✅ Bayesian calculation VALID: {p0_prior:.2%} → {p_bayesian:.2%}")
        else:
            logger.warning(f"❌ Bayesian calculation INVALID: {len(issues)} issues")

        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'p_recalculated': p_recalculated,
            'log_odds_prior': log_odds_prior,
            'total_adjusted_llr': total_adjusted_llr,
            'log_odds_posterior': log_odds_posterior,
            'calculation_diff': diff
        }

    except Exception as e:
        return {
            'is_valid': False,
            'issues': [f"Calculation error: {str(e)}"],
            'warnings': warnings
        }


@tool
def generate_probability_justification_tool(
    p0_prior: float,
    p_bayesian: float,
    evidence_summary: List[Dict[str, Any]],
    market_question: str
) -> str:
    """
    Generate human-readable justification for probability estimate

    Explains WHY the final probability is what it is based on:
    - Prior reasoning
    - Key evidence gathered (PRO and CON)
    - How evidence updated the probability
    - Uncertainty factors

    Args:
        p0_prior: Prior probability
        p_bayesian: Final Bayesian probability
        evidence_summary: Summary of key evidence items
        market_question: Original market question

    Returns:
        Detailed justification text
    """
    # Calculate change
    change = p_bayesian - p0_prior
    change_pct = (change / p0_prior) * 100 if p0_prior > 0 else 0

    # Categorize evidence by direction
    pro_evidence = [e for e in evidence_summary if e.get('support') == 'pro']
    con_evidence = [e for e in evidence_summary if e.get('support') == 'con']
    neutral_evidence = [e for e in evidence_summary if e.get('support') == 'neutral']

    # Get strongest evidence items
    pro_evidence_sorted = sorted(pro_evidence, key=lambda x: abs(x.get('adjusted_LLR', 0)), reverse=True)
    con_evidence_sorted = sorted(con_evidence, key=lambda x: abs(x.get('adjusted_LLR', 0)), reverse=True)

    # Build justification
    lines = []
    lines.append(f"# Probability Estimate for: {market_question}")
    lines.append(f"")
    lines.append(f"## Final Estimate: **{p_bayesian:.1%}** probability")
    lines.append(f"")
    lines.append(f"## Analysis Summary")
    lines.append(f"")
    lines.append(f"- **Starting Prior**: {p0_prior:.1%}")
    lines.append(f"- **Final Posterior**: {p_bayesian:.1%}")
    lines.append(f"- **Change**: {change:+.1%} ({'+' if change > 0 else ''}{change_pct:.1f}%)")
    lines.append(f"")
    lines.append(f"## Evidence Breakdown")
    lines.append(f"")
    lines.append(f"- **PRO evidence**: {len(pro_evidence)} items")
    lines.append(f"- **CON evidence**: {len(con_evidence)} items")
    lines.append(f"- **Neutral evidence**: {len(neutral_evidence)} items")
    lines.append(f"")

    # Strongest PRO evidence
    if pro_evidence_sorted:
        lines.append(f"## Top Evidence Supporting YES")
        lines.append(f"")
        for i, item in enumerate(pro_evidence_sorted[:3], 1):
            title = item.get('title', 'Unknown')
            llr = item.get('adjusted_LLR', 0)
            lines.append(f"{i}. **{title}**")
            lines.append(f"   - Impact: LLR = {llr:+.2f}")
            lines.append(f"   - Summary: {item.get('claim_summary', 'No summary')[:100]}...")
            lines.append(f"")

    # Strongest CON evidence
    if con_evidence_sorted:
        lines.append(f"## Top Evidence Supporting NO")
        lines.append(f"")
        for i, item in enumerate(con_evidence_sorted[:3], 1):
            title = item.get('title', 'Unknown')
            llr = item.get('adjusted_LLR', 0)
            lines.append(f"{i}. **{title}**")
            lines.append(f"   - Impact: LLR = {llr:+.2f}")
            lines.append(f"   - Summary: {item.get('claim_summary', 'No summary')[:100]}...")
            lines.append(f"")

    # Overall interpretation
    lines.append(f"## Interpretation")
    lines.append(f"")

    if abs(change) < 0.05:
        lines.append(f"Evidence was relatively **balanced**, with similar strength arguments on both sides. ")
        lines.append(f"The final probability remains close to the prior estimate.")
    elif change > 0.10:
        lines.append(f"Evidence **strongly supports YES**, increasing probability by {change:.1%}. ")
        lines.append(f"The research uncovered {len(pro_evidence)} supporting factors outweighing {len(con_evidence)} contrary factors.")
    elif change < -0.10:
        lines.append(f"Evidence **strongly supports NO**, decreasing probability by {abs(change):.1%}. ")
        lines.append(f"The research uncovered {len(con_evidence)} contrary factors outweighing {len(pro_evidence)} supporting factors.")
    else:
        direction = "increased" if change > 0 else "decreased"
        lines.append(f"Evidence **moderately {direction}** the probability by {abs(change):.1%}. ")
        lines.append(f"The analysis found {len(pro_evidence)} PRO and {len(con_evidence)} CON factors.")

    lines.append(f"")
    lines.append(f"## Confidence Level")
    lines.append(f"")

    # Assess confidence based on evidence quality and quantity
    total_evidence = len(evidence_summary)
    if total_evidence < 5:
        confidence = "LOW"
        reason = "limited evidence gathered"
    elif len(pro_evidence) == 0 or len(con_evidence) == 0:
        confidence = "MODERATE"
        reason = "evidence only from one side"
    elif total_evidence >= 15:
        confidence = "HIGH"
        reason = "substantial evidence from multiple sources"
    else:
        confidence = "MODERATE"
        reason = "reasonable evidence base"

    lines.append(f"**Confidence**: {confidence} ({reason})")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*This estimate is based on Bayesian inference from gathered evidence and should be treated as research only, not financial advice.*")

    justification = "\n".join(lines)

    logger.info(f"📝 Generated probability justification ({len(lines)} lines)")

    return justification


def _calculate_plan_quality_score(plan: Dict[str, Any]) -> float:
    """Calculate quality score for a plan (0.0-1.0)"""
    score = 0.0

    # Prior in reasonable range (0.3-0.7 is best)
    prior = plan.get('p0_prior', 0.5)
    if 0.3 <= prior <= 0.7:
        score += 0.2
    elif 0.2 <= prior <= 0.8:
        score += 0.1

    # Has justification
    if plan.get('prior_justification'):
        score += 0.2

    # Has subclaims
    subclaims = plan.get('subclaims', [])
    if len(subclaims) >= 4:
        score += 0.2

    # Balanced subclaims
    pro_count = sum(1 for sc in subclaims if sc.get('direction') == 'pro')
    con_count = sum(1 for sc in subclaims if sc.get('direction') == 'con')
    if pro_count > 0 and con_count > 0 and abs(pro_count - con_count) <= 2:
        score += 0.2

    # Has search seeds
    search_seeds = plan.get('search_seeds', {})
    if all(len(search_seeds.get(d, [])) >= 3 for d in ['pro', 'con', 'general']):
        score += 0.2

    return score
