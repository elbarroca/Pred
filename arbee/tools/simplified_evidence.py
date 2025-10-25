"""
Simplified Evidence Extraction - 1-10 Relevance Scoring
Replaces complex LLR calibration with simple, interpretable scoring
"""
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from arbee.models.schemas import EvidenceItem
from config.settings import Settings

logger = logging.getLogger(__name__)


async def extract_evidence_from_results(
    search_results: List[Dict[str, Any]],
    subject_name: str,
    market_question: str,
    context: str = "",
    settings: Optional[Settings] = None
) -> List[EvidenceItem]:
    """
    Extract evidence items with 1-10 relevance scoring from search results.

    Args:
        search_results: List of search results with title, url, snippet/content
        subject_name: Name of the subject being researched
        market_question: The prediction market question
        context: Additional context (e.g., "Diplo's fitness profile")
        settings: Settings object (optional)

    Returns:
        List of EvidenceItem objects with relevance scores
    """
    logger.info(f"📋 Extracting evidence from {len(search_results)} search results...")

    settings = settings or Settings()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=settings.OPENAI_API_KEY
    )

    evidence_items = []

    for result in search_results:
        try:
            # Extract basic fields
            title = result.get('title', 'N/A')
            url = result.get('url', '')
            content = result.get('content', result.get('snippet', ''))
            published_date = result.get('published_date', result.get('date', ''))

            if not content or len(content) < 20:
                logger.debug(f"Skipping result (content too short): {title[:50]}")
                continue

            # Build extraction prompt
            system_prompt = """You are an evidence analyst for prediction markets.

Your task: Extract the MOST RELEVANT FACT and make LOGICAL INFERENCES about what it means for the outcome.

**CRITICAL**: Evidence can support YES/NO even if INDIRECT. Think about implications, not just literal content.

# Relevance Scoring (1-10):

**9-10: DIRECT ANSWER** - Contains exact data needed
  - Example: "Diplo ran 5k in 24:15 at NYC race 2024" → relevance=10, NO (24:15 > 23:00)

**7-8: STRONG PROXY** - Contains closely related performance or contextual data
  - Example: "Diplo completed half marathon in 1:55" → relevance=8, YES (shows endurance)
  - Example: "Sub-23 is top 20% for age 46" → relevance=8, NO (challenging for recreational)

**5-6: USEFUL CONTEXT** - Provides meaningful baseline information
  - Example: "Diplo, 46, hosts running club events" → relevance=6, YES (baseline fitness)
  - Example: "Diplo ran LA Marathon 2023" → relevance=6, YES (commitment to running)

**3-4: WEAK RELEVANCE** - Generic benchmarks without context
  - Example: "Average 5k time for males is 34 minutes" → relevance=3, NEUTRAL (too generic)

**1-2: BARELY RELEVANT** - Off-topic
  - Example: "Running is good for health" → relevance=1, NEUTRAL

# Support Direction (MAKE INFERENCES):

- **YES**: Evidence suggests outcome WILL happen
  - Direct: "Ran 5k in 22:45" (faster than 23:00)
  - Indirect: "Ran marathon" (shows endurance → supports training ability)
  - Contextual: "Hosts Run Club" (engaged runner → baseline fitness)

- **NO**: Evidence suggests outcome will NOT happen
  - Direct: "Ran 5k in 25:30" (slower than 23:00)
  - Indirect: "Target is 5min faster than age average" (difficult achievement)
  - Contextual: "No competitive running background" (unlikely to hit competitive time)

- **NEUTRAL**: Evidence provides context but no directional signal
  - Generic stats unrelated to subject

# Is Primary Source?

- **True**: Official race results, verified performance data
- **False**: News articles, commentary, analysis

# Output Format:

Return ONLY a JSON object:

{
  "key_fact": "One sentence summary of the most relevant fact",
  "relevance_score": 1-10 (integer),
  "support_direction": "YES" | "NO" | "NEUTRAL",
  "is_primary": true | false,
  "extraction_reasoning": "Why this score? Why this direction?"
}

# Examples:

**For market: "Will Diplo run 5k in under 23 minutes?"**

Example 1 - Direct Evidence:
Source: "Diplo finished the Brooklyn 5k in 24:32 on October 1st, 2024"
Output:
{
  "key_fact": "Diplo ran 5k in 24:32 in October 2024",
  "relevance_score": 10,
  "support_direction": "NO",
  "is_primary": true,
  "extraction_reasoning": "Direct 5k time from recent race. 24:32 is 1:32 slower than 23:00 target, strongly suggesting sub-23 is unlikely without major improvement."
}

Example 2 - Indirect Performance Evidence:
Source: "Diplo completed the LA Marathon in 2023, his first marathon"
Output:
{
  "key_fact": "Diplo completed LA Marathon in 2023",
  "relevance_score": 6,
  "support_direction": "YES",
  "is_primary": false,
  "extraction_reasoning": "Marathon completion shows endurance and commitment to running training. Suggests he has the dedication to train for a specific 5k goal, supporting ability to achieve faster times."
}

Example 3 - Contextual Benchmark:
Source: "The average 5k time for 46-year-old males is 28 minutes. A good time is 22:31."
Output:
{
  "key_fact": "Average 46yo male runs 28min, good time is 22:31, target is 23:00",
  "relevance_score": 8,
  "support_direction": "NO",
  "is_primary": false,
  "extraction_reasoning": "Target of 23:00 requires being in top 20-30% for age group (5min faster than average). This is challenging for recreational runners, suggesting lower probability unless competitive background exists."
}

Example 4 - Baseline Fitness:
Source: "Diplo, 46, hosts community running events through 'Diplo's Run Club'"
Output:
{
  "key_fact": "Diplo (46) actively hosts community running events",
  "relevance_score": 5,
  "support_direction": "YES",
  "is_primary": false,
  "extraction_reasoning": "Hosting running events shows active engagement with running community and baseline fitness. Suggests recreational runner with some experience, supporting ability to train for goal."
}

Return ONLY the JSON, no additional text."""

            user_prompt = f"""Market Question: {market_question}
Subject: {subject_name}
{f'Context: {context}' if context else ''}

Source:
Title: {title}
URL: {url}
Published: {published_date or 'unknown'}

Content:
{content[:2000]}

Extract the key fact and score its relevance (1-10):"""

            system_msg = SystemMessage(content=system_prompt)
            human_msg = HumanMessage(content=user_prompt)

            response = await llm.ainvoke([system_msg, human_msg])
            response_text = response.content

            # Parse JSON
            try:
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

                data = json.loads(json_text)

                # Validate and create EvidenceItem
                evidence = EvidenceItem(
                    source_url=url,
                    published_date=published_date or None,
                    key_fact=data['key_fact'],
                    relevance_score=int(data['relevance_score']),
                    support_direction=data['support_direction'],
                    is_primary=data['is_primary'],
                    extraction_reasoning=data['extraction_reasoning']
                )

                evidence_items.append(evidence)

                logger.info(
                    f"✅ Evidence: relevance={evidence.relevance_score}/10, "
                    f"support={evidence.support_direction}, "
                    f"primary={evidence.is_primary}"
                )
                logger.debug(f"   Fact: {evidence.key_fact[:80]}")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to parse evidence extraction: {e}")
                logger.debug(f"Response was: {response_text[:300]}")
                continue

        except Exception as e:
            logger.error(f"Error extracting evidence from result: {e}")
            continue

    logger.info(f"📊 Extracted {len(evidence_items)} evidence items")
    return evidence_items


def calculate_evidence_weight(evidence: EvidenceItem) -> float:
    """
    Calculate weight for evidence based on relevance, source type, and recency.

    Formula: weight = relevance × (3 if primary else 1) × (2 if recent else 1)

    Args:
        evidence: EvidenceItem to weight

    Returns:
        Float weight value
    """
    base_weight = evidence.relevance_score

    # 3x multiplier for primary sources
    source_multiplier = 3.0 if evidence.is_primary else 1.0

    # 2x multiplier for recent sources (< 6 months)
    recency_multiplier = 1.0
    if evidence.published_date:
        try:
            pub_date = datetime.fromisoformat(evidence.published_date.replace('Z', '+00:00'))
            days_old = (datetime.now() - pub_date).days
            if days_old <= 180:  # 6 months
                recency_multiplier = 2.0
        except:
            pass

    weight = base_weight * source_multiplier * recency_multiplier

    return weight


def aggregate_evidence_to_probability(
    evidence_items: List[EvidenceItem],
    baseline_prior: float = 0.5
) -> Dict[str, Any]:
    """
    Aggregate evidence into probability estimate using weighted scoring.

    Formula:
    - Calculate weighted scores: sum(relevance × multipliers) for YES/NO
    - Normalize to probability: 0.5 + (net_score / (2 × total_weight))

    Args:
        evidence_items: List of extracted evidence
        baseline_prior: Starting probability (default 0.5)

    Returns:
        Dict with probability estimate, confidence, and breakdown
    """
    if not evidence_items:
        return {
            "probability": baseline_prior,
            "confidence": 0.0,
            "yes_weight": 0.0,
            "no_weight": 0.0,
            "neutral_weight": 0.0,
            "total_evidence": 0,
            "primary_evidence": 0
        }

    yes_weight = 0.0
    no_weight = 0.0
    neutral_weight = 0.0
    primary_count = 0

    for evidence in evidence_items:
        weight = calculate_evidence_weight(evidence)

        if evidence.support_direction == "YES":
            yes_weight += weight
        elif evidence.support_direction == "NO":
            no_weight += weight
        else:  # NEUTRAL
            neutral_weight += weight

        if evidence.is_primary:
            primary_count += 1

    total_weight = yes_weight + no_weight + neutral_weight

    if total_weight == 0:
        probability = baseline_prior
    else:
        # Net score: positive = YES evidence, negative = NO evidence
        net_score = yes_weight - no_weight

        # Normalize to probability: baseline + adjustment
        # Maximum adjustment is ±0.5 (to reach 0 or 1)
        adjustment = net_score / (2 * total_weight)
        probability = baseline_prior + adjustment

        # Clamp to [0, 1]
        probability = max(0.0, min(1.0, probability))

    # Confidence based on evidence count and quality
    confidence = min(1.0, (len(evidence_items) / 10.0) + (primary_count / 5.0))

    return {
        "probability": probability,
        "confidence": confidence,
        "yes_weight": yes_weight,
        "no_weight": no_weight,
        "neutral_weight": neutral_weight,
        "total_evidence": len(evidence_items),
        "primary_evidence": primary_count,
        "baseline_prior": baseline_prior
    }
