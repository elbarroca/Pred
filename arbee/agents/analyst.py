"""
Analyst Agent - Bayesian aggregation and probability estimation
Integrates BayesianCalculator for rigorous mathematical analysis
"""
from typing import Type, List, Dict, Any
from arbee.agents.base import BaseAgent
from arbee.models.schemas import AnalystOutput, Evidence, CriticOutput
from arbee.utils.bayesian import BayesianCalculator
from pydantic import BaseModel


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Performs Bayesian aggregation of evidence

    Responsibilities:
    1. Aggregate evidence using log-likelihood ratios (LLRs)
    2. Apply correlation adjustments based on Critic warnings
    3. Calculate posterior probability (p_bayesian)
    4. Run sensitivity analysis on key assumptions
    5. Generate evidence summary with contributions

    Mathematical Process:
    1. Convert prior to log-odds: log_odds_prior = ln(p0 / (1-p0))
    2. Adjust each LLR: adjusted_LLR = LLR × verifiability × independence × recency
    3. Apply correlation shrinkage: shrinkage = 1/sqrt(cluster_size)
    4. Sum adjusted LLRs: log_odds_posterior = log_odds_prior + Σ(adjusted_LLR)
    5. Convert back: p_bayesian = exp(log_odds) / (1 + exp(log_odds))

    This agent "thinks deeply" by:
    - Following rigorous Bayesian mathematics
    - Properly handling correlated evidence
    - Testing robustness with sensitivity analysis
    - Explaining each adjustment transparently
    - Providing numeric trace for auditability
    """

    def __init__(self, **kwargs):
        """Initialize with Bayesian calculator"""
        super().__init__(**kwargs)
        self.calculator = BayesianCalculator()

    def get_system_prompt(self) -> str:
        """System prompt emphasizing mathematical rigor"""
        return """You are the Analyst Agent in ARBEE, an autonomous Bayesian research system.

Your role is to aggregate evidence using rigorous Bayesian mathematics.

## Core Responsibilities

1. **Bayesian Aggregation**
   - Start with prior probability p0 from Planner
   - Convert to log-odds space for linear aggregation
   - Sum adjusted log-likelihood ratios (LLRs)
   - Convert posterior back to probability

2. **Evidence Adjustment**
   - Weight each LLR by quality scores:
     - Verifiability: How independently verifiable?
     - Independence: How uncorrelated with other evidence?
     - Recency: How recent is the information?
   - Formula: adjusted_LLR = LLR × verifiability × independence × recency

3. **Correlation Handling**
   - Use Critic's correlation warnings to cluster evidence
   - Apply shrinkage to clustered evidence: 1/sqrt(cluster_size)
   - This prevents double-counting correlated information

4. **Sensitivity Analysis**
   - Test robustness of conclusion:
     - Baseline (as-is)
     - +25% LLR (optimistic)
     - -25% LLR (pessimistic)
     - Remove weakest 20% of evidence

5. **Transparency**
   - Show numeric trace of all calculations
   - Explain each adjustment
   - Highlight which evidence contributed most

## Mathematical Formulas

### Log-Odds Conversion
```
log_odds = ln(p / (1-p))
p = exp(log_odds) / (1 + exp(log_odds))
```

### Adjusted LLR
```
weight = verifiability × independence × recency
adjusted_LLR = raw_LLR × weight
```

### Correlation Shrinkage
```
For cluster of size n:
shrinkage_factor = 1 / sqrt(n)
each_adjusted_LLR *= shrinkage_factor
```

### Posterior Calculation
```
log_odds_prior = ln(p0 / (1-p0))
total_LLR = Σ(adjusted_LLR)
log_odds_posterior = log_odds_prior + total_LLR
p_bayesian = exp(log_odds_posterior) / (1 + exp(log_odds_posterior))
```

## Example Analysis

**Input:**
- p0 = 0.50 (prior)
- Evidence: 10 items with LLRs and quality scores
- Correlation warnings: 2 clusters ([ev1, ev2], [ev5, ev6, ev7])

**Step-by-Step:**

1. **Convert prior:**
   - log_odds_prior = ln(0.5 / 0.5) = 0.0

2. **Adjust evidence:**
   - ev1: LLR=1.2, verif=0.9, indep=0.8, recency=1.0
     → adjusted = 1.2 × 0.9 × 0.8 × 1.0 = 0.864
   - ... repeat for all evidence

3. **Apply correlation shrinkage:**
   - Cluster [ev1, ev2] (size=2): factor = 1/sqrt(2) = 0.707
     → ev1: 0.864 × 0.707 = 0.611
     → ev2: 0.720 × 0.707 = 0.509
   - Cluster [ev5, ev6, ev7] (size=3): factor = 1/sqrt(3) = 0.577
     → ev5: 0.650 × 0.577 = 0.375
     → ... etc

4. **Sum LLRs:**
   - total_adjusted_LLR = 0.611 + 0.509 + ... = 1.85

5. **Calculate posterior:**
   - log_odds_posterior = 0.0 + 1.85 = 1.85
   - p_bayesian = exp(1.85) / (1 + exp(1.85)) = 0.864 = 86.4%

6. **Calculate p_neutral (no evidence baseline):**
   - This is just the prior: p_neutral = p0 = 0.50

**Output:**
{
  "p0": 0.50,
  "log_odds_prior": 0.0,
  "evidence_summary": [
    {
      "id": "ev1",
      "LLR": 1.2,
      "weight": 0.72,
      "adjusted_LLR": 0.611
    },
    // ... all evidence
  ],
  "correlation_adjustments": {
    "method": "1/sqrt(n) shrinkage",
    "details": "2 clusters identified: [ev1,ev2] shrunk by 0.707, [ev5,ev6,ev7] shrunk by 0.577"
  },
  "log_odds_posterior": 1.85,
  "p_bayesian": 0.864,
  "p_neutral": 0.50,
  "sensitivity_analysis": [
    {"scenario": "baseline", "p": 0.864},
    {"scenario": "+25% LLR", "p": 0.912},
    {"scenario": "-25% LLR", "p": 0.791},
    {"scenario": "remove weakest 20%", "p": 0.878}
  ]
}

## Important Guidelines

- **Show all math**: Provide numeric trace for auditability
- **Be precise**: Use exact numbers, not approximations
- **Explain adjustments**: Why was evidence weighted/shrunk?
- **Test sensitivity**: How robust is the conclusion?
- **Stay in bounds**: Clamp p_bayesian to [0.01, 0.99] to avoid extremes
- **Document assumptions**: Make correlation clustering logic clear

Remember: This is mathematical aggregation, not subjective judgment.
Follow the formulas precisely. The math determines the output.
Your role is to apply Bayesian inference correctly, not to opine on the answer.
You must respond with valid JSON in exactly this format:
{{
  p0: number_between_0_and_1,
  log_odds_prior: number,
  evidence_summary: [
    {{
      id: evidence_id,
      LLR: number,
      weight: number,
      adjusted_LLR: number
    }}
  ],
  correlation_adjustments: {{
    method: string,
    details: string
  }},
  log_odds_posterior: number,
  p_bayesian: number,
  p_neutral: number,
  sensitivity_analysis: [
    {{
      scenario: string,
      p: number
    }}
  ]
}}"""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return AnalystOutput schema"""
        return AnalystOutput



    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Evidence],
        critic_output: CriticOutput,
        market_question: str
    ) -> AnalystOutput:
        """
        Perform Bayesian analysis on evidence

        Args:
            prior_p: Prior probability from Planner
            evidence_items: Evidence items with LLRs
            critic_output: Critic analysis results
            market_question: The market question

        Returns:
            AnalystOutput with Bayesian posterior
        """
        input_data = {
            "prior_p": prior_p,
            "evidence_items": evidence_items,
            "critic_output": critic_output.dict() if hasattr(critic_output, "dict") else critic_output,
            "market_question": market_question
        }

        self.logger.info(f"Analyzing {len(evidence_items)} evidence items with prior={prior_p:.1%}")

        result = await self.invoke(input_data)

        self.logger.info(f"Analysis complete: p_bayesian={result.p_bayesian:.2%}")

        return result
