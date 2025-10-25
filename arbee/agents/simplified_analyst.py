"""
Simplified Analyst Agent - Weighted Scoring Instead of Complex LLR
Aggregates evidence into probability estimate using simple, interpretable math
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from arbee.models.schemas import (
    SimplifiedAnalystOutput,
    EvidenceItem,
    SubjectProfile
)
from arbee.tools.simplified_evidence import (
    calculate_evidence_weight,
    aggregate_evidence_to_probability
)
from config.settings import Settings

logger = logging.getLogger(__name__)


class SimplifiedAnalystAgent:
    """
    Simplified Analyst - replaces complex Bayesian LLR with weighted scoring.

    Formula:
    - weight = relevance × (3 if primary else 1) × (2 if recent else 1)
    - net_score = sum(YES weights) - sum(NO weights)
    - probability = 0.5 + (net_score / (2 × total_weight))
    - Clamped to [0, 1]
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        self.settings = settings or Settings()
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.settings.OPENAI_API_KEY
        )
        logger.info(f"SimplifiedAnalystAgent initialized with {model_name}")

    async def analyze(
        self,
        market_question: str,
        subject_profile: SubjectProfile,
        benchmark_evidence: List[EvidenceItem],
        specific_evidence: List[EvidenceItem],
        baseline_prior: float = 0.5
    ) -> SimplifiedAnalystOutput:
        """
        Analyze evidence and produce probability estimate.

        Args:
            market_question: The prediction market question
            subject_profile: Profile of the subject
            benchmark_evidence: General benchmark/context evidence
            specific_evidence: Specific evidence about the subject
            baseline_prior: Starting probability (default 0.5)

        Returns:
            SimplifiedAnalystOutput with probability estimate and reasoning
        """
        logger.info("🧮 Starting simplified analysis...")
        logger.info(f"   Benchmark evidence: {len(benchmark_evidence)} items")
        logger.info(f"   Specific evidence: {len(specific_evidence)} items")
        logger.info(f"   Baseline prior: {baseline_prior:.2%}")

        # Combine all evidence
        all_evidence = benchmark_evidence + specific_evidence

        if not all_evidence:
            logger.warning("⚠️ No evidence available, returning baseline prior")
            return SimplifiedAnalystOutput(
                probability=baseline_prior,
                confidence=0.0,
                yes_evidence_weight=0.0,
                no_evidence_weight=0.0,
                neutral_evidence_weight=0.0,
                total_evidence_count=0,
                primary_evidence_count=0,
                baseline_prior=baseline_prior,
                reasoning="No evidence available, using baseline prior of 50%",
                top_yes_evidence=[],
                top_no_evidence=[],
                sensitivity_range={"low": baseline_prior, "high": baseline_prior}
            )

        # Calculate weighted scores
        logger.info("\n📊 Calculating evidence weights...")

        yes_weights = []
        no_weights = []
        neutral_weights = []

        yes_evidence_items = []
        no_evidence_items = []

        for evidence in all_evidence:
            weight = calculate_evidence_weight(evidence)

            if evidence.support_direction == "YES":
                yes_weights.append(weight)
                yes_evidence_items.append({
                    "key_fact": evidence.key_fact,
                    "relevance": evidence.relevance_score,
                    "weight": weight,
                    "is_primary": evidence.is_primary,
                    "source_url": evidence.source_url
                })
            elif evidence.support_direction == "NO":
                no_weights.append(weight)
                no_evidence_items.append({
                    "key_fact": evidence.key_fact,
                    "relevance": evidence.relevance_score,
                    "weight": weight,
                    "is_primary": evidence.is_primary,
                    "source_url": evidence.source_url
                })
            else:  # NEUTRAL
                neutral_weights.append(weight)

        # If we have evidence but it's all/mostly neutral, try contextual inference
        if len(all_evidence) >= 5 and len(yes_weights) == 0 and len(no_weights) == 0:
            logger.info("\n🔍 All evidence is NEUTRAL - attempting contextual inference...")
            inferred_evidence = await self.infer_from_neutral_context(
                market_question=market_question,
                subject_profile=subject_profile,
                neutral_evidence=all_evidence,
                baseline_prior=baseline_prior
            )

            if inferred_evidence:
                logger.info(f"   Generated {len(inferred_evidence)} contextual inferences")
                for inf in inferred_evidence:
                    weight = inf['weight']
                    if inf['direction'] == 'YES':
                        yes_weights.append(weight)
                        yes_evidence_items.append({
                            "key_fact": inf['reasoning'],
                            "relevance": inf['confidence'] * 10,  # Scale to 1-10
                            "weight": weight,
                            "is_primary": False,
                            "source_url": "contextual_inference"
                        })
                    else:  # NO
                        no_weights.append(weight)
                        no_evidence_items.append({
                            "key_fact": inf['reasoning'],
                            "relevance": inf['confidence'] * 10,
                            "weight": weight,
                            "is_primary": False,
                            "source_url": "contextual_inference"
                        })

        yes_total = sum(yes_weights)
        no_total = sum(no_weights)
        neutral_total = sum(neutral_weights)
        total_weight = yes_total + no_total + neutral_total

        logger.info(f"   YES weight: {yes_total:.2f} ({len(yes_weights)} items)")
        logger.info(f"   NO weight: {no_total:.2f} ({len(no_weights)} items)")
        logger.info(f"   NEUTRAL weight: {neutral_total:.2f} ({len(neutral_weights)} items)")
        logger.info(f"   Total weight: {total_weight:.2f}")

        # Calculate probability
        if total_weight == 0:
            probability = baseline_prior
            logger.warning("⚠️ Total weight is zero, using baseline prior")
        else:
            # Net score: positive = YES evidence, negative = NO evidence
            net_score = yes_total - no_total

            # Normalize to probability
            # Max adjustment is ±0.5 (to reach 0 or 1)
            adjustment = net_score / (2 * total_weight)
            probability = baseline_prior + adjustment

            # Clamp to [0, 1]
            probability = max(0.0, min(1.0, probability))

            logger.info(f"\n🎯 Probability calculation:")
            logger.info(f"   Net score: {net_score:+.2f}")
            logger.info(f"   Adjustment: {adjustment:+.2f}")
            logger.info(f"   Final probability: {probability:.2%}")

        # Calculate confidence based on evidence quantity and quality
        primary_count = sum(1 for e in all_evidence if e.is_primary)

        # Confidence: more evidence + primary sources = higher confidence
        evidence_confidence = min(1.0, len(all_evidence) / 20.0)  # 20+ items = max
        primary_confidence = min(1.0, primary_count / 5.0)  # 5+ primary = max

        # Penalize if too many neutrals
        neutral_penalty = 0.0
        if total_weight > 0:
            neutral_ratio = neutral_total / total_weight
            if neutral_ratio > 0.7:  # More than 70% neutral
                neutral_penalty = 0.3

        confidence = (0.6 * evidence_confidence + 0.4 * primary_confidence) - neutral_penalty
        confidence = max(0.0, min(1.0, confidence))

        logger.info(f"   Confidence: {confidence:.2%}")
        logger.info(f"   Primary sources: {primary_count}/{len(all_evidence)}")

        # Sensitivity analysis: what if we're ±25% wrong on weights?
        low_probability = baseline_prior + (net_score * 0.75) / (2 * total_weight) if total_weight > 0 else baseline_prior
        high_probability = baseline_prior + (net_score * 1.25) / (2 * total_weight) if total_weight > 0 else baseline_prior
        low_probability = max(0.0, min(1.0, low_probability))
        high_probability = max(0.0, min(1.0, high_probability))

        # Sort evidence by weight for top items
        yes_evidence_items.sort(key=lambda x: x['weight'], reverse=True)
        no_evidence_items.sort(key=lambda x: x['weight'], reverse=True)

        # Generate reasoning using LLM
        reasoning = await self.generate_reasoning(
            market_question=market_question,
            subject_profile=subject_profile,
            probability=probability,
            confidence=confidence,
            yes_evidence=yes_evidence_items[:5],
            no_evidence=no_evidence_items[:5],
            yes_total=yes_total,
            no_total=no_total,
            neutral_total=neutral_total
        )

        output = SimplifiedAnalystOutput(
            probability=probability,
            confidence=confidence,
            yes_evidence_weight=yes_total,
            no_evidence_weight=no_total,
            neutral_evidence_weight=neutral_total,
            total_evidence_count=len(all_evidence),
            primary_evidence_count=primary_count,
            baseline_prior=baseline_prior,
            reasoning=reasoning,
            top_yes_evidence=yes_evidence_items[:5],
            top_no_evidence=no_evidence_items[:5],
            sensitivity_range={
                "low": low_probability,
                "high": high_probability
            }
        )

        logger.info("\n" + "="*80)
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Final probability: {probability:.2%}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Sensitivity range: {low_probability:.2%} - {high_probability:.2%}")

        return output

    async def infer_from_neutral_context(
        self,
        market_question: str,
        subject_profile: SubjectProfile,
        neutral_evidence: List[EvidenceItem],
        baseline_prior: float
    ) -> List[Dict[str, Any]]:
        """
        Generate contextual inferences when all evidence is neutral.
        Analyzes relationships between evidence items to produce directional signals.
        """

        # Prepare evidence summary
        evidence_facts = "\n".join([
            f"- {e.key_fact} (relevance={e.relevance_score})"
            for e in neutral_evidence[:15]  # Top 15 most relevant
        ])

        system_prompt = """You are a Bayesian inference engine. Your task is to analyze NEUTRAL evidence and make logical inferences about directional probability.

Even when evidence doesn't directly answer the question, you can infer implications:
- Performance in related events → capability assessment
- Benchmark comparisons → difficulty assessment
- Background/experience → likelihood estimation

Generate 2-5 contextual inferences with directional signals (YES/NO) and confidence (0.0-1.0).

Output JSON array:
[
  {
    "direction": "YES" | "NO",
    "reasoning": "Clear explanation of the inference",
    "confidence": 0.0-1.0 (how strong is this inference),
    "weight": 1-20 (relevance × confidence × 10)
  }
]

Examples:

Market: "Will Diplo run 5k in under 23 minutes?"
Evidence: "Diplo ran LA Marathon 2023", "Hosts Run Club events", "Average 46yo runs 28min", "Good male time is 22:31"

Inferences:
[
  {
    "direction": "YES",
    "reasoning": "Marathon completion demonstrates endurance and running commitment, suggesting capability to train for fast 5k",
    "confidence": 0.5,
    "weight": 10
  },
  {
    "direction": "YES",
    "reasoning": "Hosting Run Club events indicates active engagement with running community and baseline fitness",
    "confidence": 0.4,
    "weight": 8
  },
  {
    "direction": "NO",
    "reasoning": "Target time (23:00) is 5 minutes faster than age average (28:00), requiring top 20% performance - challenging for recreational runner",
    "confidence": 0.7,
    "weight": 14
  },
  {
    "direction": "NO",
    "reasoning": "No evidence of competitive running background or recent race times suggests recreational level, making sub-23 unlikely",
    "confidence": 0.6,
    "weight": 12
  }
]

Return ONLY the JSON array."""

        user_prompt = f"""Market Question: {market_question}

Subject Profile:
- Name: {subject_profile.entity_name}
- Age: {subject_profile.key_facts.get('age', 'Unknown')}
- Background: {subject_profile.baseline_capabilities or 'Unknown'}

Neutral Evidence (no direct answer found):
{evidence_facts}

Generate contextual inferences with directional signals:"""

        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(content=user_prompt)

        try:
            response = await self.llm.ainvoke([system_msg, human_msg])
            response_text = response.content

            # Parse JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            inferences = json.loads(json_text)

            logger.info(f"\n💡 Generated {len(inferences)} contextual inferences:")
            for inf in inferences:
                logger.info(f"   {inf['direction']}: {inf['reasoning'][:80]}... (weight={inf['weight']})")

            return inferences

        except Exception as e:
            logger.error(f"Failed to generate contextual inferences: {e}")
            return []

    async def generate_reasoning(
        self,
        market_question: str,
        subject_profile: SubjectProfile,
        probability: float,
        confidence: float,
        yes_evidence: List[Dict[str, Any]],
        no_evidence: List[Dict[str, Any]],
        yes_total: float,
        no_total: float,
        neutral_total: float
    ) -> str:
        """Generate human-readable reasoning for the probability estimate"""

        system_prompt = """You are a prediction analyst. Generate a clear, concise explanation (3-5 sentences) for why the probability estimate makes sense.

Focus on:
1. What the evidence shows (YES vs NO balance)
2. Quality of evidence (primary sources, relevance)
3. Key uncertainties or gaps
4. Overall confidence level

Be direct and factual. Don't hedge excessively."""

        user_prompt = f"""Market Question: {market_question}

Subject Profile:
- Name: {subject_profile.entity_name}
- Type: {subject_profile.entity_type}
- Background: {subject_profile.baseline_capabilities or 'Unknown'}
- Key Facts: {json.dumps(subject_profile.key_facts, indent=2)}

Evidence Summary:
- YES evidence weight: {yes_total:.2f} ({len(yes_evidence)} items)
- NO evidence weight: {no_total:.2f} ({len(no_evidence)} items)
- NEUTRAL evidence weight: {neutral_total:.2f}

Top YES Evidence:
{json.dumps([{'fact': e['key_fact'][:100], 'relevance': e['relevance'], 'primary': e['is_primary']} for e in yes_evidence[:3]], indent=2)}

Top NO Evidence:
{json.dumps([{'fact': e['key_fact'][:100], 'relevance': e['relevance'], 'primary': e['is_primary']} for e in no_evidence[:3]], indent=2)}

Probability: {probability:.2%}
Confidence: {confidence:.2%}

Write a clear explanation for this probability estimate:"""

        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(content=user_prompt)

        try:
            response = await self.llm.ainvoke([system_msg, human_msg])
            reasoning = response.content.strip()
            logger.info(f"\n💭 Generated reasoning: {reasoning[:200]}...")
            return reasoning
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return f"Probability {probability:.1%} based on {yes_total:.1f} YES weight vs {no_total:.1f} NO weight. Confidence {confidence:.1%}."
