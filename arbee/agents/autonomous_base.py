"""Autonomous ReAct base agent with deterministic memory- and tool-aware control flow."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage as GraphAnyMessage, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from config.system_constants import (
    AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
    AUTO_QUERY_SIMILAR_MARKETS_LIMIT,
    AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT,
    MEMORY_QUERY_TIMEOUT_SECONDS,
)
from config.settings import Settings


logger = logging.getLogger(__name__)


class ThoughtStep(BaseModel):
    """Single reasoning step captured for transparency and auditing."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thought: str = Field(description="Agent inner monologue excerpt")
    reasoning: str = Field(description="Expanded reasoning for the thought")
    action_plan: Optional[str] = Field(default=None, description="Intended next action")


class ToolCallRecord(BaseModel):
    """Structured record of a tool invocation."""

    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


class MemoryItem(BaseModel):
    """Item retrieved from long-term memory search."""

    key: str
    content: Any
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """Shared state for autonomous agents in the ReAct workflow."""

    messages: Annotated[List[GraphAnyMessage], add_messages]
    reasoning_trace: Annotated[List[ThoughtStep], "Chain of thought steps"]
    tool_calls: Annotated[List[ToolCallRecord], "Tools used during reasoning"]
    memory_accessed: Annotated[List[MemoryItem], "Memories retrieved from long-term storage"]
    intermediate_results: Annotated[Dict[str, Any], "Temporary data during reasoning"]
    final_output: Annotated[Optional[BaseModel], "Final structured result"]
    next_action: Annotated[Literal["continue", "end", "escalate"], "What to do next"]
    iteration_count: Annotated[int, "Number of reasoning loops completed"]
    max_iterations: Annotated[int, "Maximum iterations before forcing termination"]
    task_description: Annotated[str, "High-level description of what agent should accomplish"]
    task_input: Annotated[Dict[str, Any], "Input data for the task"]


class AutonomousReActAgent(ABC):
    """Reusable autonomous agent base implementing the ReAct pattern with strict safeguards."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 20,
        store: Optional[BaseStore] = None,
        *,
        auto_extend_iterations: bool = True,
        iteration_extension: int = 5,
        max_iteration_cap: int = 50,
        recursion_limit: Optional[int] = None,
        llm_timeout: float = 60.0,
        agent_timeout: float = 600.0,
        enable_memory_tracking: bool = True,
        enable_query_deduplication: bool = True,
        enable_url_blocking: bool = True,
        enable_circuit_breakers: bool = True,
        enable_auto_memory_query: bool = True,
    ) -> None:
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.store = store
        self.auto_extend_iterations = auto_extend_iterations
        self.iteration_extension = iteration_extension
        self.max_iteration_cap = max_iteration_cap
        self.recursion_limit = recursion_limit
        self.llm_timeout = llm_timeout
        self.agent_timeout = agent_timeout
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_query_deduplication = enable_query_deduplication
        self.enable_url_blocking = enable_url_blocking
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_memory_query = enable_auto_memory_query

        if self.enable_auto_memory_query:
            if self.store is None:
                raise ValueError("Memory store is required when enable_auto_memory_query is True.")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.llm_timeout,
        )

        self.tools = self.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        self.tool_node = ToolNode(self.tools) if self.tools else None

        self.stats: Dict[str, Any] = {
            "total_invocations": 0,
            "successful_completions": 0,
            "max_iterations_reached": 0,
            "total_tool_calls": 0,
            "total_memory_accesses": 0,
            "average_iterations": 0.0,
        }

        logger.info(
            "%s ready (tools=%d, max_iterations=%d)",
            self.__class__.__name__,
            len(self.tools),
            self.max_iterations,
        )

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt guiding the agent."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return available tools for the agent."""

    def _record_event(self, state: AgentState, category: str, detail: str) -> None:
        diagnostics = state.setdefault("diagnostics", [])
        diagnostics.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": state.get("iteration_count", 0),
                "category": category,
                "detail": detail,
            }
        )

    def _recent_tool_names(self, messages: List[BaseMessage], window: int) -> List[str]:
        names: List[str] = []
        for message in messages[-window:]:
            if isinstance(message, AIMessage) and message.tool_calls:
                names.extend(call.get("name", "unknown") for call in message.tool_calls)
        return names

    def _recent_queries(self, intermediate: Dict[str, Any], window: int) -> List[str]:
        queries = intermediate.get("last_N_queries", []) if intermediate else []
        return list(queries[-window:])

    def _message_text(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        parts.append(str(block.get("text", "")))
                    elif block_type == "json":
                        parts.append(json.dumps(block.get("json", {})))
            if parts:
                return " ".join(parts).strip()
        return str(content)

    def _preview(self, message: BaseMessage, limit: int = 240) -> str:
        text = self._message_text(message)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _format_tool_args(self, args: Any, limit: int = 140) -> str:
        serialized = json.dumps(args, default=str)
        return serialized if len(serialized) <= limit else f"{serialized[:limit]}..."

    def _parse_tool_result(self, tool_message: ToolMessage) -> Any:
        artifact = getattr(tool_message, "artifact", None)
        if artifact is not None:
            if isinstance(artifact, (dict, list, str)):
                return artifact
            if hasattr(artifact, "model_dump"):
                return artifact.model_dump()
            if hasattr(artifact, "__dict__"):
                return vars(artifact)
        content = self._message_text(tool_message)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return None

    def _sanitize_message_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        sanitized: List[BaseMessage] = []
        pending_tool_ids: List[str] = []
        for message in messages:
            if isinstance(message, AIMessage):
                sanitized.append(message)
                pending_tool_ids = [
                    call.get("id")
                    for call in (message.tool_calls or [])
                    if isinstance(call, dict) and call.get("id")
                ]
            elif isinstance(message, ToolMessage):
                tool_call_id = getattr(message, "tool_call_id", None)
                if pending_tool_ids:
                    if tool_call_id and tool_call_id in pending_tool_ids:
                        pending_tool_ids.remove(tool_call_id)
                        sanitized.append(message)
                    elif not tool_call_id and pending_tool_ids:
                        pending_tool_ids.pop(0)
                    sanitized.append(message)
                else:
                    has_recent_tool_call = any(
                        isinstance(prev_msg, AIMessage) and prev_msg.tool_calls
                        for prev_msg in sanitized[-5:]
                    )
                    if has_recent_tool_call:
                        sanitized.append(message)
                if isinstance(message, HumanMessage):
                    pending_tool_ids = []
        return sanitized

    def _build_memory_context(self, state: AgentState) -> str:
        intermediate = state.get("intermediate_results", {})
        context_parts: List[str] = []
        last_queries = self._recent_queries(intermediate, 5)
        if last_queries:
            context_parts.append(
                "MEMORY REMINDER: recent queries already issued:\n"
                + "\n".join(f"- {query}" for query in last_queries)
            )
        blocked = intermediate.get("blocked_urls", set())
        if blocked:
            blocked_list = sorted(list(blocked))[:3]
            context_parts.append(
                "BLOCKED URLS: further extraction should be skipped:\n"
                + "\n".join(f"- {url[:80]}" for url in blocked_list)
            )
        return "\n\n".join(context_parts)

    async def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message: ToolMessage,
    ) -> None:
        return

    @abstractmethod
    async def is_task_complete(self, state: AgentState) -> bool:
        """Return True when the agent has satisfied its completion criteria."""

    @abstractmethod
    async def extract_final_output(self, state: AgentState) -> BaseModel:
        """Produce final structured output from the terminal agent state."""

    async def query_memory(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        assert self.store is not None, "Memory store is not configured."
        assert limit > 0, "limit must be positive"
        results = await self.store.asearch(
            query=query,
            namespace=namespace or self.__class__.__name__,
            limit=limit,
        )
        memory_items = [
            MemoryItem(
                key=result.key,
                content=result.value,
                relevance_score=getattr(result, "score", 1.0),
                metadata=getattr(result, "metadata", {}),
            )
            for result in results
        ]
        self.stats["total_memory_accesses"] += 1
        return memory_items

    async def store_memory(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        assert self.store is not None, "Memory store is not configured."
        await self.store.aput(namespace or self.__class__.__name__, key, value, metadata or {})
        return True

    async def _fetch_similar_markets(
        self,
        market_question: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        from arbee.tools.memory_search import search_similar_markets_tool

        payload = {"market_question": market_question, "limit": limit}
        return await asyncio.wait_for(search_similar_markets_tool.ainvoke(payload), timeout=timeout)

    async def _fetch_historical_evidence(
        self,
        topic: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        from arbee.tools.memory_search import search_historical_evidence_tool

        payload = {"topic": topic, "limit": limit}
        return await asyncio.wait_for(
            search_historical_evidence_tool.ainvoke(payload), timeout=timeout
        )

    async def _fetch_successful_strategies(
        self,
        query: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        assert self.store is not None, "Memory store is not configured."
        search_results = await asyncio.wait_for(
            self.store.asearch(("strategies",), query=query, limit=limit), timeout=timeout
        )
        strategies: List[Dict[str, Any]] = []
        for item in search_results:
            payload = item.value or {}
            if isinstance(payload, dict):
                effectiveness = payload.get("effectiveness", 0.0)
                if effectiveness >= 0.7:
                    strategies.append(
                        {
                            "description": payload.get("description", ""),
                            "effectiveness": effectiveness,
                            "strategy_type": payload.get("strategy_type", "unknown"),
                        }
                    )
        return strategies

    async def _query_memory_at_start(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        state: AgentState,
    ) -> str:
        if not self.enable_memory_tracking or self.store is None:
            return ""

        market_question = (
            task_input.get("market_question") or task_input.get("question") or task_description
        )
        similar_markets = await self._fetch_similar_markets(
            market_question, AUTO_QUERY_SIMILAR_MARKETS_LIMIT, MEMORY_QUERY_TIMEOUT_SECONDS
        )
        historical_evidence = await self._fetch_historical_evidence(
            market_question[:100],
            AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
            MEMORY_QUERY_TIMEOUT_SECONDS,
        )
        strategies = await self._fetch_successful_strategies(
            market_question, AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT, MEMORY_QUERY_TIMEOUT_SECONDS
        )

        memory_items: List[MemoryItem] = []
        context_lines: List[str] = []

        if similar_markets:
            context_lines.append(
                "SIMILAR MARKETS:\n"
                + "\n".join(
                    f"- {item.get('question', 'Unknown')} (prior={item.get('prior', 'N/A')}, outcome={item.get('outcome', 'N/A')})"
                    for item in similar_markets[:3]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=item.get("id", "unknown"),
                    content=item,
                    relevance_score=item.get("score", 0.8),
                    metadata={"type": "similar_market"},
                )
                for item in similar_markets
            )

        if historical_evidence:
            context_lines.append(
                "HISTORICAL EVIDENCE:\n"
                + "\n".join(
                    f"- {item.get('title', 'Unknown')} (LLR={item.get('llr', 0.0):.2f}, support={item.get('support', 'unknown')})"
                    for item in historical_evidence[:3]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=item.get("id", "unknown"),
                    content=item,
                    relevance_score=item.get("relevance_score", 0.7),
                    metadata={"type": "historical_evidence"},
                )
                for item in historical_evidence
            )

        if strategies:
            context_lines.append(
                "SUCCESSFUL STRATEGIES:\n"
                + "\n".join(
                    f"- [{item['strategy_type']}] {item['description']} (effectiveness={item['effectiveness']:.1%})"
                    for item in strategies[:3]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=f"strategy_{hash(item['description'])}",
                    content=item,
                    relevance_score=item["effectiveness"],
                    metadata={"type": "successful_strategy"},
                )
                for item in strategies
            )

        if memory_items:
            state["memory_accessed"] = memory_items
            self._record_event(state, "memory_context", f"retrieved {len(memory_items)} items")
            header = "=" * 59
            context = (
                f"{header}\nMEMORY CONTEXT\n{header}\n" + "\n\n".join(context_lines) + f"\n{header}"
            )
            return context

    def _detect_pre_iteration_stall(self, state: AgentState) -> bool:
        if state["iteration_count"] < 5:
            return False
        recent_tools = self._recent_tool_names(state.get("messages", []), 10)
        if len(recent_tools) >= 5 and len(set(recent_tools[-5:])) == 1:
            tool_name = recent_tools[-1]
            self._record_event(state, "stall_detected", f"Repeat tool {tool_name}")
            state["_forced_stop"] = True
            return True
        return False

    async def agent_node(self, state: AgentState) -> AgentState:
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        if self._detect_pre_iteration_stall(state):
            return state

        system_message = SystemMessage(content=self.get_system_prompt())
        task_message = HumanMessage(
            content=f"Task: {state['task_description']}\n\nInput: {state['task_input']}"
        )
        state_messages = state.get("messages", [])
        # Temporarily disable message sanitization to debug
        # if state_messages:
        #     sanitized = self._sanitize_message_history(state_messages)
        #     if len(sanitized) != len(state_messages):
        #         state["messages"] = sanitized
        #     state_messages = state["messages"]
        memory_context = self._build_memory_context(state)
        messages: List[BaseMessage] = [system_message, task_message]
        if memory_context:
            messages.append(SystemMessage(content=memory_context))
        # For tool-using agents, only include the most recent conversation turn
        # to avoid message pairing issues
        if self.tools:
            # Find the last AI message with tool calls
            last_ai_with_tools = None
            for msg in reversed(state_messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    last_ai_with_tools = msg
                    break
            if last_ai_with_tools:
                # Include the AI message and any subsequent tool messages
                ai_index = state_messages.index(last_ai_with_tools)
                relevant_messages = state_messages[ai_index:]
                messages.extend(relevant_messages)
            else:
                # No tool calls in history, just keep recent messages
                recent_messages = state_messages[-10:] if len(state_messages) > 10 else state_messages
                messages.extend(recent_messages)
        else:
            # For non-tool agents, keep more history
            recent_messages = state_messages[-20:] if len(state_messages) > 20 else state_messages
            messages.extend(recent_messages)

        response = await self.llm_with_tools.ainvoke(messages)
        state["messages"] = state.get("messages", []) + [response]

        response_text = self._message_text(response)
        if response_text or response.tool_calls:
            state.setdefault("reasoning_trace", []).append(
                ThoughtStep(
                    thought=response_text[:1000],
                    reasoning=response_text[1000:2000],
                    action_plan="Tool calls scheduled" if response.tool_calls else "Task complete",
                )
            )
            if response.tool_calls:
                call_names = [call.get("name", "unknown") for call in response.tool_calls]
                state.setdefault("diagnostics", []).append(
                    {
                        "iteration": state["iteration_count"],
                        "tool_calls": call_names,
                        "response_text": response_text[1000:2000],
                        "memory_context": memory_context[:100] if memory_context else None,
                    }
                )
            return state

    def _detect_query_loop(self, state: AgentState) -> bool:
        if state["iteration_count"] < 5:
            return False
        intermediate = state.get("intermediate_results", {})
        recent = self._recent_queries(intermediate, 5)
        if len(recent) == 5 and len(set(recent)) <= 2:
            self._record_event(state, "query_loop", str(recent))
            return True
        return False

    def _detect_tool_loop(self, state: AgentState, threshold: int) -> bool:
        recent_tools = self._recent_tool_names(state.get("messages", []), threshold)
        if len(recent_tools) >= threshold:
            unique_tools = len(set(recent_tools[-threshold:]))
            if unique_tools <= 2:
                counts = Counter(recent_tools[-threshold:])
                self._record_event(state, "tool_loop", json.dumps(counts))
                return True
        return False

    def _detect_validation_loop(self, state: AgentState) -> bool:
        recent_tools = self._recent_tool_names(state.get("messages", []), 6)
        if len(recent_tools) < 4:
            return False
        validation_calls = [name for name in recent_tools[-6:] if "validate" in name.lower()]
        if len(validation_calls) < 4:
            return False
        prior_found = False
        for message in reversed(state.get("messages", [])[-20:]):
            if isinstance(message, ToolMessage) and "validate_prior_tool" in str(
                getattr(message, "name", "")
            ):
                tool_result = self._parse_tool_result(message)
                if isinstance(tool_result, dict) and "prior_p" in tool_result:
                    state.setdefault("intermediate_results", {})["p0_prior"] = tool_result[
                        "prior_p"
                    ]
                    state["intermediate_results"]["prior_validated"] = tool_result.get(
                        "is_valid", False
                    )
                    state["intermediate_results"]["prior_justification"] = tool_result.get(
                        "justification", ""
                    )
                    prior_found = True
                    break
        detail = "prior captured" if prior_found else "prior missing"
        self._record_event(state, "validation_loop", detail)
        return True

    def _detect_progress_stall(self, state: AgentState) -> bool:
        if state["iteration_count"] < 5:
            return False
        current_results = state.get("intermediate_results", {})
        current_hash = hash(json.dumps(current_results, sort_keys=True, default=str))
        prev_hash = state.get("_results_hash")
        state["_results_hash"] = current_hash
        if prev_hash == current_hash:
            state["_no_progress_count"] = state.get("_no_progress_count", 0) + 1
            if state["_no_progress_count"] >= 3:
                self._record_event(state, "progress_stall", "no change in intermediate_results")
                return True
        else:
            state["_no_progress_count"] = 0
        return False

    def _manage_iteration_budget(self, state: AgentState) -> None:
        if state["iteration_count"] < state["max_iterations"]:
            return
        if self.auto_extend_iterations and state["max_iterations"] < self.max_iteration_cap:
            new_limit = min(
                state["max_iterations"] + self.iteration_extension, self.max_iteration_cap
            )
            state["max_iterations"] = new_limit
            self._record_event(state, "iteration_extended", str(new_limit))
        else:
            state.setdefault("_forced_stop_reason", "max_iterations")
            self.stats["max_iterations_reached"] += 1

    async def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        if state.get("_forced_stop"):
            self._record_event(state, "forced_stop", "pre-iteration stall")
            return "end"
        if self._detect_query_loop(state):
            return "end"
        if self._detect_tool_loop(state, 6):
            return "end"
        last_message = state["messages"][-1] if state.get("messages") else None
        if last_message and getattr(last_message, "tool_calls", None):
            return "tools"
        if self._detect_tool_loop(state, 5):
            return "end"
        if self._detect_validation_loop(state):
            return "end"
        if self._detect_progress_stall(state):
            return "end"
        self._manage_iteration_budget(state)
        if state.get("_forced_stop_reason") == "max_iterations":
            return "end"
        if state["iteration_count"] >= state["max_iterations"] and not self.auto_extend_iterations:
            return "end"
        if await self.is_task_complete(state):
            return "end"
        return "end"

    async def create_reasoning_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.agent_node)

        if self.tool_node:

            async def tools_wrapper(state: AgentState) -> AgentState:
                prior_messages = state.get("messages", [])
                if not prior_messages or not isinstance(prior_messages[-1], AIMessage):
                    self._record_event(state, "tool_execution_skipped", "missing AI message")
                    return state
                tool_result = await self.tool_node.ainvoke(state)
                tool_messages: List[BaseMessage] = (
                    tool_result.get("messages", []) if tool_result else []
                )
                if not tool_messages:
                    self._record_event(state, "tool_execution_skipped", "tool returned no messages")
                    return state
                last_ai = prior_messages[-1]
                call_lookup = {}
                if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                    for call in last_ai.tool_calls:
                        call_lookup[call.get("id")] = call
                state["messages"] = prior_messages + tool_messages
                for message in tool_messages:
                    if isinstance(message, ToolMessage):
                        call_meta = call_lookup.get(message.tool_call_id, {})
                        tool_name = getattr(message, "name", call_meta.get("name", "unknown_tool"))
                        tool_args = call_meta.get("args", {})
                        record = ToolCallRecord(
                            tool_name=tool_name,
                            tool_input=tool_args,
                            tool_output=self._message_text(message),
                        )
                        state.setdefault("tool_calls", []).append(record)
                        self.stats["total_tool_calls"] += 1
                        tool_result_payload = self._parse_tool_result(message)
                        if tool_name == "estimate_prior_with_base_rates_tool" and isinstance(
                            tool_result_payload, dict
                        ):
                            state.setdefault("intermediate_results", {})[
                                "prior_reasoning"
                            ] = tool_result_payload
                        if tool_name == "bayesian_calculate_tool" and isinstance(
                            tool_result_payload, dict
                        ):
                            state.setdefault("intermediate_results", {}).update(
                                {
                                    "p0": tool_result_payload.get("p0"),
                                    "p_bayesian": tool_result_payload.get("p_bayesian"),
                                    "log_odds_prior": tool_result_payload.get("log_odds_prior"),
                                    "log_odds_posterior": tool_result_payload.get(
                                        "log_odds_posterior"
                                    ),
                                    "p_neutral": tool_result_payload.get("p_neutral", 0.5),
                                    "evidence_summary": tool_result_payload.get(
                                        "evidence_summary", []
                                    ),
                                    "correlation_adjustments": tool_result_payload.get(
                                        "correlation_adjustments", {}
                                    ),
                                }
                            )
                        if tool_name == "sensitivity_analysis_tool" and isinstance(
                            tool_result_payload, dict
                        ):
                            state.setdefault("intermediate_results", {})[
                                "sensitivity_analysis"
                            ] = tool_result_payload
                        if tool_name == "validate_prior_tool" and isinstance(
                            tool_result_payload, dict
                        ):
                            intermediate = state.setdefault("intermediate_results", {})
                            intermediate["p0_prior"] = tool_result_payload.get(
                                "prior_p", tool_result_payload.get("p0")
                            )
                            intermediate["prior_validated"] = tool_result_payload.get(
                                "is_valid", False
                            )
                            intermediate["prior_justification"] = tool_result_payload.get(
                                "justification", ""
                            )
                        await self.handle_tool_message(state, tool_name, tool_args, message)
                return state

            workflow.add_node("tools", tools_wrapper)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools" if self.tool_node else END,
                "end": END,
            },
        )
        if self.tool_node:
            workflow.add_edge("tools", "agent")
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    async def run(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        *,
        max_iterations: Optional[int] = None,
    ) -> BaseModel:
        self.stats["total_invocations"] += 1
        effective_max_iterations = max_iterations or self.max_iterations
        if self.auto_extend_iterations:
            effective_max_iterations = min(effective_max_iterations, self.max_iteration_cap)
        initial_state: AgentState = {
            "messages": [],
            "reasoning_trace": [],
            "tool_calls": [],
            "memory_accessed": [],
            "intermediate_results": {},
            "final_output": None,
            "next_action": "continue",
            "iteration_count": 0,
            "max_iterations": effective_max_iterations,
            "task_description": task_description,
            "task_input": task_input,
        }
        from config.system_constants import ENABLE_AUTO_MEMORY_QUERY_DEFAULT

        enable_auto_query = getattr(
            self, "enable_auto_memory_query", ENABLE_AUTO_MEMORY_QUERY_DEFAULT
        )
        if enable_auto_query and self.store is not None:
            memory_context = await self._query_memory_at_start(
                task_description, task_input, initial_state
            )
            if memory_context:
                initial_state.setdefault("messages", []).append(
                    HumanMessage(content=memory_context)
                )

        app = await self.create_reasoning_graph()
        default_config = {
            "configurable": {
                "thread_id": f"{self.__class__.__name__}-{datetime.utcnow().timestamp()}",
            }
        }
        merged_config = default_config
        if config:
            merged_config = {**default_config, **config}
            if "configurable" in config:
                merged_config["configurable"] = {
                    **default_config.get("configurable", {}),
                    **config["configurable"],
                }
        if "recursion_limit" not in merged_config:
            merged_config["recursion_limit"] = self.recursion_limit or max(
                60, effective_max_iterations * 5
            )

        final_state = await asyncio.wait_for(
            app.ainvoke(initial_state, merged_config), timeout=self.agent_timeout
        )
        if final_state.get("error"):
            raise RuntimeError(f"Agent terminated with error: {final_state['error']}")
        if not await self.is_task_complete(final_state):
            raise RuntimeError("Agent terminated without satisfying completion criteria.")

        output = await self.extract_final_output(final_state)
        self.stats["successful_completions"] += 1
        iterations = final_state["iteration_count"]
        completions = self.stats["successful_completions"]
        prev_avg = self.stats["average_iterations"]
        self.stats["average_iterations"] = (
            (prev_avg * (completions - 1)) + iterations
        ) / completions
        self.logger.info("%s finished in %d iterations", self.__class__.__name__, iterations)
        return output

    def get_stats(self) -> Dict[str, Any]:
        success_rate = (
            self.stats["successful_completions"] / self.stats["total_invocations"]
            if self.stats["total_invocations"]
            else 0.0
        )
        return {**self.stats, "success_rate": success_rate}
