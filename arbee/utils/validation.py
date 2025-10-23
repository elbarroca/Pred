"""
Validation utilities for POLYSEER
Includes Chain-of-Thought validation helpers
"""
from typing import Any
from pydantic import BaseModel


def validate_cot_output(
    output: BaseModel,
    min_length: int = 100,
    min_steps: int = 3,
    strict: bool = True
) -> None:
    """
    Validate that Chain-of-Thought outputs are meaningful and complete

    Args:
        output: Agent output (PlannerOutput, AnalystOutput, CriticOutput, etc.)
        min_length: Minimum character length for text CoT fields (default 100)
        min_steps: Minimum number of steps for list CoT fields (default 3)
        strict: If True, raise ValueError on validation failure. If False, log warning.

    Raises:
        ValueError: If CoT fields are too short or missing (when strict=True)

    Examples:
        >>> from arbee.models.schemas import PlannerOutput
        >>> output = await planner.plan("Will Trump win 2024?")
        >>> validate_cot_output(output, min_length=150)  # Validates reasoning_trace
        >>>
        >>> from arbee.models.schemas import AnalystOutput
        >>> output = await analyst.analyze(...)
        >>> validate_cot_output(output, min_steps=5)  # Validates calculation_steps
    """
    errors = []

    # Check for reasoning_trace (Planner, Researcher)
    if hasattr(output, 'reasoning_trace'):
        trace = output.reasoning_trace
        if not trace or not isinstance(trace, str):
            errors.append("reasoning_trace is missing or not a string")
        elif len(trace) < min_length:
            errors.append(
                f"reasoning_trace too short: {len(trace)} chars (minimum {min_length}). "
                f"Expected detailed step-by-step reasoning."
            )
        elif not any(keyword in trace.lower() for keyword in ['step 1', 'first', 'initially']):
            errors.append(
                "reasoning_trace does not appear to contain step-by-step reasoning. "
                "Expected explicit steps like 'Step 1:', 'Step 2:', etc."
            )

    # Check for calculation_steps (Analyst)
    if hasattr(output, 'calculation_steps'):
        steps = output.calculation_steps
        if not steps or not isinstance(steps, list):
            errors.append("calculation_steps is missing or not a list")
        elif len(steps) < min_steps:
            errors.append(
                f"calculation_steps has only {len(steps)} steps (minimum {min_steps}). "
                f"Expected detailed step-by-step calculation explanation."
            )
        elif any(not step or len(step.strip()) < 20 for step in steps):
            errors.append(
                "calculation_steps contains steps that are too short or empty. "
                "Each step should be a meaningful explanation (minimum 20 chars)."
            )

    # Check for analysis_process (Critic)
    if hasattr(output, 'analysis_process'):
        process = output.analysis_process
        if not process or not isinstance(process, str):
            errors.append("analysis_process is missing or not a string")
        elif len(process) < min_length:
            errors.append(
                f"analysis_process too short: {len(process)} chars (minimum {min_length}). "
                f"Expected detailed analysis methodology."
            )
        elif not any(keyword in process.lower() for keyword in ['step 1', 'first', 'initially']):
            errors.append(
                "analysis_process does not appear to contain step-by-step analysis. "
                "Expected explicit steps like 'Step 1:', 'Step 2:', etc."
            )

    # Check for search_strategy (Researcher - if it exists in schema)
    if hasattr(output, 'search_strategy'):
        strategy = output.search_strategy
        if not strategy or not isinstance(strategy, str):
            errors.append("search_strategy is missing or not a string")
        elif len(strategy) < min_length // 2:  # Less strict for search strategy
            errors.append(
                f"search_strategy too short: {len(strategy)} chars (minimum {min_length//2}). "
                f"Expected explanation of search approach."
            )

    # Raise or log errors
    if errors:
        error_message = (
            f"Chain-of-Thought validation failed for {type(output).__name__}:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )
        if strict:
            raise ValueError(error_message)
        else:
            import logging
            logging.warning(error_message)


def validate_evidence_quality(
    evidence: Any,
    min_verifiability: float = 0.3,
    min_independence: float = 0.3,
    require_url: bool = True,
    require_date: bool = True
) -> None:
    """
    Validate evidence item quality scores and required fields

    Args:
        evidence: Evidence object to validate
        min_verifiability: Minimum acceptable verifiability score (0-1)
        min_independence: Minimum acceptable independence score (0-1)
        require_url: If True, URL must be present
        require_date: If True, published_date must be present

    Raises:
        ValueError: If evidence quality is below thresholds

    Example:
        >>> from arbee.models.schemas import Evidence
        >>> evidence = Evidence(...)
        >>> validate_evidence_quality(evidence, min_verifiability=0.5)
    """
    errors = []

    # Check quality scores
    if hasattr(evidence, 'verifiability_score'):
        if evidence.verifiability_score < min_verifiability:
            errors.append(
                f"verifiability_score too low: {evidence.verifiability_score:.2f} "
                f"(minimum {min_verifiability:.2f})"
            )

    if hasattr(evidence, 'independence_score'):
        if evidence.independence_score < min_independence:
            errors.append(
                f"independence_score too low: {evidence.independence_score:.2f} "
                f"(minimum {min_independence:.2f})"
            )

    # Check required fields
    if require_url and hasattr(evidence, 'url'):
        if not evidence.url or evidence.url == "unknown":
            errors.append("URL is missing or 'unknown'")

    if require_date and hasattr(evidence, 'published_date'):
        if not evidence.published_date:
            errors.append("published_date is missing")

    # Check claim summary quality
    if hasattr(evidence, 'claim_summary'):
        summary = evidence.claim_summary
        if not summary or len(summary.strip()) < 50:
            errors.append(
                f"claim_summary too short: {len(summary)} chars (minimum 50). "
                "Evidence claims should be specific and detailed."
            )

    if errors:
        error_message = (
            f"Evidence quality validation failed:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )
        raise ValueError(error_message)


def validate_probability(p: float, name: str = "probability") -> None:
    """
    Validate that a probability value is in valid range [0, 1]

    Args:
        p: Probability value to validate
        name: Name of the probability (for error messages)

    Raises:
        ValueError: If probability is out of range

    Example:
        >>> validate_probability(0.75, "prior_p")
        >>> validate_probability(1.5, "prior_p")  # Raises ValueError
    """
    if not isinstance(p, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(p)}")

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"{name} must be in range [0.0, 1.0], got {p}"
        )


def validate_llr(llr: float, max_magnitude: float = 5.0) -> None:
    """
    Validate that a log-likelihood ratio is reasonable

    Args:
        llr: Log-likelihood ratio to validate
        max_magnitude: Maximum acceptable magnitude (default 5.0)

    Raises:
        ValueError: If LLR magnitude is unreasonably high

    Example:
        >>> validate_llr(1.5)  # OK
        >>> validate_llr(10.0)  # Raises ValueError (too extreme)
    """
    if not isinstance(llr, (int, float)):
        raise ValueError(f"LLR must be numeric, got {type(llr)}")

    if abs(llr) > max_magnitude:
        raise ValueError(
            f"LLR magnitude too large: {llr} (max {max_magnitude}). "
            f"LLRs above {max_magnitude} are extremely rare and should be reviewed."
        )
