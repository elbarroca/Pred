"""
Pydantic schemas for POLYSEER agent outputs
All schemas follow the JSON structures defined in CLAUDE.MD
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, date


# ============================================================================
# PLANNER AGENT SCHEMAS
# ============================================================================

class Subclaim(BaseModel):
    """A subclaim decomposed from the main market question"""
    id: str = Field(..., description="Unique identifier for subclaim")
    text: str = Field(..., description="The subclaim text")
    direction: Literal["pro", "con"] = Field(..., description="Whether this supports YES or NO")


class SearchSeeds(BaseModel):
    """Search query seeds for evidence gathering"""
    pro: List[str] = Field(default_factory=list, description="Queries supporting YES")
    con: List[str] = Field(default_factory=list, description="Queries supporting NO")
    general: List[str] = Field(default_factory=list, description="Neutral/contextual queries")


class PlannerOutput(BaseModel):
    """Output from the Planner Agent"""
    market_slug: str = Field(..., description="Unique market identifier")
    market_question: str = Field(..., description="The prediction market question")
    p0_prior: float = Field(..., ge=0.0, le=1.0, description="Initial prior probability")
    prior_justification: str = Field(..., description="Reasoning for the prior")
    subclaims: List[Subclaim] = Field(..., min_length=4, max_length=10)
    key_variables: List[str] = Field(..., description="Critical factors affecting outcome")
    search_seeds: SearchSeeds
    decision_criteria: List[str] = Field(..., description="What would resolve the question")


# ============================================================================
# RESEARCHER AGENT SCHEMAS
# ============================================================================

class Evidence(BaseModel):
    """A single piece of evidence from research"""
    subclaim_id: str = Field(..., description="References Subclaim.id")
    title: str = Field(..., description="Article/source title")
    url: str = Field(..., description="Full URL to source")
    published_date: date = Field(..., description="Publication date YYYY-MM-DD")
    source_type: Literal["primary", "high_quality_secondary", "secondary", "weak"]
    claim_summary: str = Field(..., max_length=500, description="Key claim extracted")
    support: Literal["pro", "con", "neutral"] = Field(..., description="Direction of evidence")
    verifiability_score: float = Field(..., ge=0.0, le=1.0)
    independence_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)
    estimated_LLR: float = Field(..., description="Log-likelihood ratio estimate")
    extraction_notes: str = Field(..., description="Context and caveats")

    @field_validator('estimated_LLR')
    @classmethod
    def validate_llr_range(cls, v: float, info) -> float:
        """Validate LLR is within calibrated ranges"""
        source_type = info.data.get('source_type')
        if source_type == 'primary' and abs(v) > 3.0:
            raise ValueError(f"Primary source LLR should be ±1-3, got {v}")
        elif source_type in ['high_quality_secondary', 'secondary'] and abs(v) > 1.0:
            raise ValueError(f"Secondary source LLR should be ±0.1-1.0, got {v}")
        return v


class ResearcherOutput(BaseModel):
    """Output from Researcher Agents (aggregated)"""
    evidence_items: List[Evidence] = Field(..., max_length=30)
    total_pro_count: int = Field(default=0)
    total_con_count: int = Field(default=0)
    research_timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# CRITIC AGENT SCHEMAS
# ============================================================================

class CorrelationWarning(BaseModel):
    """Warning about correlated evidence"""
    cluster: List[str] = Field(..., description="List of evidence IDs in cluster")
    note: str = Field(..., description="Explanation of correlation")


class CriticOutput(BaseModel):
    """Output from the Critic Agent"""
    duplicate_clusters: List[List[str]] = Field(
        default_factory=list,
        description="Groups of duplicate/near-duplicate evidence IDs"
    )
    missing_topics: List[str] = Field(
        default_factory=list,
        description="Important angles not covered"
    )
    over_represented_sources: List[str] = Field(
        default_factory=list,
        description="Sources appearing too frequently"
    )
    correlation_warnings: List[CorrelationWarning] = Field(default_factory=list)
    follow_up_search_seeds: List[str] = Field(
        default_factory=list,
        description="Additional queries to fill gaps"
    )


# ============================================================================
# ANALYST AGENT SCHEMAS
# ============================================================================

class EvidenceSummaryItem(BaseModel):
    """Summary of a single evidence item's contribution"""
    id: str = Field(..., description="Evidence ID or subclaim_id")
    LLR: float = Field(..., description="Original log-likelihood ratio")
    weight: float = Field(..., ge=0.0, le=1.0, description="Quality weight applied")
    adjusted_LLR: float = Field(..., description="Final LLR after adjustments")


class CorrelationAdjustment(BaseModel):
    """Details on how correlation was handled"""
    method: str = Field(..., description="E.g., 'shrinkage', 'cluster_averaging'")
    details: str = Field(..., description="Explanation of adjustments made")


class SensitivityScenario(BaseModel):
    """Alternative probability under different assumptions"""
    scenario: str = Field(..., description="E.g., '+25% LLR', 'remove_weakest_sources'")
    p: float = Field(..., ge=0.0, le=1.0, description="Resulting probability")


class AnalystOutput(BaseModel):
    """Output from the Analyst Agent (Bayesian aggregator)"""
    p0: float = Field(..., ge=0.0, le=1.0, description="Prior probability")
    log_odds_prior: float = Field(..., description="ln(p0 / (1-p0))")
    evidence_summary: List[EvidenceSummaryItem]
    correlation_adjustments: CorrelationAdjustment
    log_odds_posterior: float = Field(..., description="After evidence aggregation")
    p_bayesian: float = Field(..., ge=0.0, le=1.0, description="Final probability estimate")
    p_neutral: float = Field(..., ge=0.0, le=1.0, description="Uncertainty-adjusted estimate")
    sensitivity_analysis: List[SensitivityScenario] = Field(default_factory=list)

    @field_validator('p_bayesian', 'p_neutral')
    @classmethod
    def clamp_extremes(cls, v: float) -> float:
        """Prevent probabilities too close to 0 or 1"""
        if v < 0.01:
            return 0.01
        if v > 0.99:
            return 0.99
        return v


# ============================================================================
# ARBITRAGE DETECTOR SCHEMAS
# ============================================================================

class PlatformSide(BaseModel):
    """Specific side of a trade on a platform"""
    platform: Literal["polymarket", "kalshi", "calci"]
    market_id: str = Field(..., description="Market identifier")
    outcome: Literal["YES", "NO"] = Field(..., description="Which outcome to bet on")
    price: float = Field(..., ge=0.0, le=1.0, description="Price for this outcome")
    stake: float = Field(..., ge=0.0, description="Amount to stake on this side")


class ArbitrageOpportunity(BaseModel):
    """A potential arbitrage opportunity"""
    arbitrage_type: Literal["mispricing", "cross_platform"] = Field(
        ...,
        description="Type: mispricing (single-sided based on Bayesian) or cross_platform (opposite sides, guaranteed)"
    )

    # Single platform fields (for mispricing arbitrage)
    market_id: Optional[str] = Field(None, description="Market identifier from provider")
    provider: Optional[Literal["polymarket", "kalshi", "calci"]] = None
    price: Optional[float] = Field(None, ge=0.0, le=1.0, description="Current market price")
    implied_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    edge: Optional[float] = Field(None, description="p_bayesian - implied_probability")

    # Cross-platform fields (for true arbitrage)
    platform_pair: Optional[List[str]] = Field(
        None,
        description="List of platforms involved in cross-platform arb"
    )
    side_a: Optional[PlatformSide] = Field(
        None,
        description="First side of the arbitrage (e.g., YES on Polymarket)"
    )
    side_b: Optional[PlatformSide] = Field(
        None,
        description="Second side of the arbitrage (e.g., NO on Kalshi)"
    )
    total_cost: Optional[float] = Field(
        None,
        description="Total cost to enter both sides (with fees)"
    )
    guaranteed_profit: Optional[float] = Field(
        None,
        description="Guaranteed profit per dollar (for cross-platform arb)"
    )

    # Common fields
    transaction_costs: float = Field(default=0.0, ge=0.0)
    slippage_estimate: float = Field(default=0.0, ge=0.0)
    expected_value_per_dollar: float = Field(..., description="EV after costs")
    kelly_fraction: float = Field(..., ge=0.0, le=1.0, description="Optimal bet fraction")
    suggested_stake: float = Field(..., ge=0.0, description="Dollar amount to stake")
    trade_rationale: str = Field(..., description="Why this is/isn't a good trade")


class ArbitrageDetectorOutput(BaseModel):
    """Output from Arbitrage Detector"""
    opportunities: List[ArbitrageOpportunity] = Field(default_factory=list)
    best_opportunity: Optional[ArbitrageOpportunity] = None
    total_expected_value: float = Field(default=0.0)
    disclaimer: str = Field(default="NOT FINANCIAL ADVICE")


# ============================================================================
# REPORTER AGENT SCHEMAS
# ============================================================================

class TopDriver(BaseModel):
    """A key factor influencing the forecast"""
    direction: Literal["pro", "con"]
    summary: str = Field(..., max_length=200)
    strength: Literal["strong", "moderate", "weak"]


class ReporterOutput(BaseModel):
    """Final output from Reporter Agent"""
    market_question: str
    p_bayesian: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: tuple[float, float] = Field(
        ...,
        description="(lower, upper) bounds"
    )
    top_pro_drivers: List[TopDriver] = Field(..., max_length=3)
    top_con_drivers: List[TopDriver] = Field(..., max_length=3)
    arbitrage_summary: str
    next_steps: List[str] = Field(default_factory=list)
    tldr: str = Field(..., max_length=300, description="1-2 sentence summary")
    executive_summary: str = Field(
        ...,
        min_length=200,
        max_length=600,
        description="Markdown summary"
    )
    full_json: Dict[str, Any] = Field(..., description="Complete data package")
    disclaimer: str = Field(default="NOT FINANCIAL ADVICE")


# ============================================================================
# COMBINED WORKFLOW STATE
# ============================================================================

class WorkflowState(BaseModel):
    """State passed through LangGraph workflow"""
    # Input
    market_url: str
    market_question: str
    providers: List[Literal["polymarket", "kalshi", "calci"]] = ["polymarket", "kalshi"]
    bankroll: float = 10000.0
    edge_threshold: float = 0.02

    # Agent outputs (populated during workflow)
    planner_output: Optional[PlannerOutput] = None
    researcher_output: Optional[ResearcherOutput] = None
    critic_output: Optional[CriticOutput] = None
    analyst_output: Optional[AnalystOutput] = None
    arbitrage_output: Optional[ArbitrageDetectorOutput] = None
    reporter_output: Optional[ReporterOutput] = None

    # Metadata
    workflow_id: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    created_at: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = Field(default_factory=list)
