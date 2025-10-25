"""
Simplified Researcher Agent - 3-Phase Adaptive Search
PHASE 1: PROFILING - Understand WHO/WHAT the subject is
PHASE 2: BENCHMARKING - Establish what's NORMAL
PHASE 3: EVIDENCE GATHERING - Find SPECIFIC recent data

Key improvements:
- Agent generates its OWN search queries based on reasoning
- Stops when confidence > 0.7 (questions answered), not iteration count
- No pre-defined search seeds - adaptive based on what it learns
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from arbee.models.schemas import (
    SimplifiedResearcherOutput,
    SubjectProfile,
    EvidenceItem,
    ResearchPhase
)
from arbee.tools.search import web_search_tool
from arbee.tools.simplified_evidence import extract_evidence_from_results
from config.settings import Settings

logger = logging.getLogger(__name__)


class SimplifiedResearcherAgent:
    """
    3-Phase Adaptive Researcher

    Unlike the old system that executed pre-defined search seeds,
    this agent REASONS about what to search next based on what it learns.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_searches: int = 10
    ):
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_searches = max_searches

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.settings.OPENAI_API_KEY
        )

        logger.info(f"SimplifiedResearcherAgent initialized (max_searches={max_searches})")

    def get_phase_prompt(self, phase: str, context: Dict[str, Any]) -> str:
        """Get the system prompt for the current research phase"""

        if phase == "profiling":
            return f"""You are in PHASE 1: PROFILING

Your task: Learn WHO/WHAT the subject is.

Subject: {context['subject_name']}
Entity Type: {context['entity_type']}
Market Question: {context['market_question']}

Generate a search query to build a basic profile of {context['subject_name']}.

Focus on:
- Who/what is this subject? (age, profession, background if person)
- What is their baseline capability? (relevant experience, training, history)
- Any relevant context for the market question

Output JSON:
{{
  "reasoning": "Why I need this information",
  "search_query": "actual Google-style query",
  "expected_info": "What facts I hope to extract"
}}

Example for "Diplo" (person, sports context):
{{
  "reasoning": "I need to know who Diplo is and their fitness background to assess running capability",
  "search_query": "Diplo age profession fitness running background",
  "expected_info": "Diplo's age, occupation, any athletic history or running experience"
}}

Return ONLY the JSON, no extra text."""

        elif phase == "benchmarking":
            return f"""You are in PHASE 2: BENCHMARKING

Your task: Establish what's NORMAL for this type of outcome.

Subject Profile: {json.dumps(context.get('subject_profile', {}), indent=2)}
Market Question: {context['market_question']}

Now that you know about the subject, generate search queries to find BENCHMARK data.

Focus on:
- What's a typical/average outcome for this type of event?
- What's considered good/competitive/excellent?
- How does the subject's profile compare to typical performers?

Output JSON:
{{
  "reasoning": "Why these benchmarks matter",
  "search_queries": ["query1", "query2"],
  "expected_info": "What performance levels to expect"
}}

Example for "46 year old runner, 5k time":
{{
  "reasoning": "Need to establish what 5k times are normal for a 46 year old male to contextualize if sub-23 is achievable",
  "search_queries": [
    "average 5k time 46 year old male",
    "competitive 5k time masters runner"
  ],
  "expected_info": "Typical 5k times for 40-50 age group, what qualifies as competitive vs recreational"
}}

Return ONLY the JSON, no extra text."""

        elif phase == "evidence_gathering":
            return f"""You are in PHASE 3: EVIDENCE GATHERING

Your task: Find SPECIFIC, RECENT data about the subject.

Subject Profile: {json.dumps(context.get('subject_profile', {}), indent=2)}
Benchmark Data: {context.get('benchmark_summary', 'Not yet established')}
Market Question: {context['market_question']}

Generate targeted queries for RECENT, SPECIFIC evidence about {context['subject_name']}.

Focus on:
- Actual performance data (race results, times, scores)
- Recent activity (last 6 months preferred)
- Training/preparation for the specific goal
- Primary sources when possible (official results vs news articles)

Output JSON:
{{
  "reasoning": "Why these sources would have primary data",
  "search_queries": ["query1", "query2", "query3"],
  "stop_criteria": "When have I gathered enough evidence?"
}}

Example for "Diplo 5k times":
{{
  "reasoning": "Need Diplo's actual race performance data to predict 5k capability",
  "search_queries": [
    "Diplo 5k race results 2024",
    "Diplo half marathon time",
    "Diplo Run Club results"
  ],
  "stop_criteria": "When I find 3+ pieces of primary performance data (actual race times) OR have searched 10 times"
}}

Return ONLY the JSON, no extra text."""

        return ""

    async def execute_search_phase(
        self,
        phase: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a search for the current phase and return results"""

        logger.info(f"📍 PHASE: {phase.upper()}")

        # Get LLM to generate search query/queries
        prompt = self.get_phase_prompt(phase, context)
        system_msg = SystemMessage(content=prompt)
        human_msg = HumanMessage(content="Generate the search query JSON:")

        response = await self.llm.ainvoke([system_msg, human_msg])
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

            query_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM query response: {e}")
            logger.error(f"Response was: {response_text[:300]}")
            # Fallback to simple query
            query_data = {
                "reasoning": "Fallback search",
                "search_query": context['subject_name'],
                "search_queries": [context['subject_name']],
                "expected_info": "Basic information"
            }

        # Execute searches
        queries = query_data.get("search_queries", [query_data.get("search_query")])
        if isinstance(queries, str):
            queries = [queries]

        all_results = []
        for query in queries:
            logger.info(f"🔍 Search query: '{query}'")
            logger.info(f"   Reasoning: {query_data.get('reasoning', 'N/A')}")

            # Execute actual web search
            results = await web_search_tool.ainvoke({"query": query, "max_results": 5})

            logger.info(f"   Found {len(results)} results")
            all_results.extend(results)

        return {
            "reasoning": query_data.get("reasoning", ""),
            "queries": queries,
            "expected_info": query_data.get("expected_info", ""),
            "results": all_results
        }

    async def extract_profile_from_results(
        self,
        results: List[Dict[str, Any]],
        subject_name: str,
        entity_type: str
    ) -> SubjectProfile:
        """Extract subject profile from search results using LLM"""

        logger.info(f"📋 Extracting profile for {subject_name}...")

        # Prepare results summary for LLM
        results_text = "\n\n".join([
            f"Source: {r.get('title', 'N/A')}\n{r.get('snippet', '')}"
            for r in results[:10]  # Top 10 results
        ])

        prompt = f"""Extract a profile for {subject_name} from these search results.

Subject: {subject_name}
Type: {entity_type}

Search Results:
{results_text}

Extract:
1. Key facts (age, profession, background, relevant experience)
2. Baseline capabilities (for the context of the market question)

Output JSON:
{{
  "key_facts": {{
    "age": "value or unknown",
    "profession": "value or unknown",
    "other_key_fact": "value"
  }},
  "baseline_capabilities": "Brief assessment of their ability/experience"
}}

Example for Diplo:
{{
  "key_facts": {{
    "age": "46",
    "profession": "DJ and music producer",
    "running_background": "Runs 'Diplo's Run Club' events"
  }},
  "baseline_capabilities": "Recreational runner who hosts community running events, some endurance experience from half marathon"
}}

Return ONLY JSON."""

        system_msg = SystemMessage(content=prompt)
        human_msg = HumanMessage(content="Extract the profile JSON:")

        response = await self.llm.ainvoke([system_msg, human_msg])
        response_text = response.content

        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            profile_data = json.loads(json_text)

            profile = SubjectProfile(
                entity_name=subject_name,
                entity_type=entity_type,
                key_facts=profile_data.get("key_facts", {}),
                baseline_capabilities=profile_data.get("baseline_capabilities")
            )

            logger.info(f"✅ Profile extracted: {len(profile.key_facts)} facts")
            return profile

        except Exception as e:
            logger.error(f"Failed to extract profile: {e}")
            # Return minimal profile
            return SubjectProfile(
                entity_name=subject_name,
                entity_type=entity_type,
                key_facts={"note": "Failed to extract profile"},
                baseline_capabilities="Unknown"
            )

    async def research(
        self,
        subject_name: str,
        entity_type: str,
        market_question: str,
        research_questions: List[str]
    ) -> SimplifiedResearcherOutput:
        """
        Execute 3-phase adaptive research

        Returns:
            SimplifiedResearcherOutput with profile, benchmarks, and evidence
        """
        logger.info(f"🚀 Starting 3-phase research for: {subject_name}")
        logger.info(f"   Questions to answer: {len(research_questions)}")

        # Initialize state
        output = SimplifiedResearcherOutput(
            subject_profile=None,
            benchmark_evidence=[],
            specific_evidence=[],
            research_phase=ResearchPhase(
                phase="profiling",
                questions_answered=[],
                questions_remaining=research_questions.copy(),
                confidence=0.0
            ),
            search_queries_used=[],
            total_searches=0
        )

        # PHASE 1: PROFILING
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: PROFILING - Understanding the subject")
        logger.info("="*80)

        profile_context = {
            "subject_name": subject_name,
            "entity_type": entity_type,
            "market_question": market_question
        }

        profile_search = await self.execute_search_phase("profiling", profile_context)
        output.search_queries_used.append({
            "phase": "profiling",
            "reasoning": profile_search["reasoning"],
            "queries": profile_search["queries"]
        })
        output.total_searches += len(profile_search["queries"])

        # Extract profile
        profile = await self.extract_profile_from_results(
            profile_search["results"],
            subject_name,
            entity_type
        )
        output.subject_profile = profile
        output.research_phase.questions_answered.append(f"Who is {subject_name}")
        output.research_phase.confidence += 0.2
        output.research_phase.phase = "benchmarking"

        logger.info(f"✅ Profile complete - Confidence: {output.research_phase.confidence:.2f}")

        # PHASE 2: BENCHMARKING
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: BENCHMARKING - Establishing baselines")
        logger.info("="*80)

        benchmark_context = {
            "subject_name": subject_name,
            "entity_type": entity_type,
            "market_question": market_question,
            "subject_profile": profile.model_dump()
        }

        benchmark_search = await self.execute_search_phase("benchmarking", benchmark_context)
        output.search_queries_used.append({
            "phase": "benchmarking",
            "reasoning": benchmark_search["reasoning"],
            "queries": benchmark_search["queries"]
        })
        output.total_searches += len(benchmark_search["queries"])

        # Extract benchmark evidence with 1-10 scoring
        logger.info("📊 Extracting benchmark evidence...")
        benchmark_evidence = await extract_evidence_from_results(
            search_results=benchmark_search["results"],
            subject_name=subject_name,
            market_question=market_question,
            context=f"Understanding normal/baseline performance for: {market_question}",
            settings=self.settings
        )
        output.benchmark_evidence = benchmark_evidence
        logger.info(f"   Extracted {len(benchmark_evidence)} benchmark evidence items")

        output.research_phase.questions_answered.append("Established benchmarks")
        output.research_phase.confidence += 0.2
        output.research_phase.phase = "evidence_gathering"

        logger.info(f"✅ Benchmarking complete - Confidence: {output.research_phase.confidence:.2f}")

        # PHASE 3: EVIDENCE GATHERING
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: EVIDENCE GATHERING - Finding specific data")
        logger.info("="*80)

        evidence_context = {
            "subject_name": subject_name,
            "entity_type": entity_type,
            "market_question": market_question,
            "subject_profile": profile.model_dump(),
            "benchmark_summary": f"Searched for benchmarks with {len(benchmark_search['queries'])} queries"
        }

        evidence_search = await self.execute_search_phase("evidence_gathering", evidence_context)
        output.search_queries_used.append({
            "phase": "evidence_gathering",
            "reasoning": evidence_search["reasoning"],
            "queries": evidence_search["queries"]
        })
        output.total_searches += len(evidence_search["queries"])

        # Extract specific evidence with 1-10 scoring
        logger.info("🎯 Extracting specific evidence...")
        specific_evidence = await extract_evidence_from_results(
            search_results=evidence_search["results"],
            subject_name=subject_name,
            market_question=market_question,
            context=f"Profile: {profile.baseline_capabilities or 'Unknown'}",
            settings=self.settings
        )
        output.specific_evidence = specific_evidence
        logger.info(f"   Extracted {len(specific_evidence)} specific evidence items")

        # Log evidence summary
        if specific_evidence:
            primary_count = sum(1 for e in specific_evidence if e.is_primary)
            avg_relevance = sum(e.relevance_score for e in specific_evidence) / len(specific_evidence)
            logger.info(f"   Primary sources: {primary_count}/{len(specific_evidence)}")
            logger.info(f"   Average relevance: {avg_relevance:.1f}/10")

        output.research_phase.questions_answered.append("Gathered specific evidence")
        output.research_phase.confidence += 0.3
        output.research_phase.phase = "complete"

        logger.info(f"✅ Evidence gathering complete - Confidence: {output.research_phase.confidence:.2f}")

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 RESEARCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Total searches: {output.total_searches}")
        logger.info(f"Final confidence: {output.research_phase.confidence:.2%}")
        logger.info(f"Questions answered: {len(output.research_phase.questions_answered)}")

        return output
