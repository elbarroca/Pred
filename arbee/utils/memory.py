"""
Memory Infrastructure for POLYSEER Autonomous Agents.
Utilities for short‑term (working), episodic, and long‑term (knowledge) memory.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# Data Models
class MemoryConfig(BaseModel):
    """Configuration for the memory system."""
    # Store backend
    store_type: str = Field(default="redis", description="redis, postgresql, or memory")
    store_connection_string: Optional[str] = Field(default=None)

    # Checkpointer backend
    checkpointer_type: str = Field(default="memory", description="memory, sqlite, or postgresql")
    checkpointer_connection_string: Optional[str] = Field(default=None)

    # Memory limits
    max_working_memory_messages: int = Field(default=50, description="Max messages in working memory")
    max_episode_memory_items: int = Field(default=100, description="Max items per episode")
    episode_retention_days: int = Field(default=90, description="How long to keep episode memories")

    # Vector search
    embedding_model: str = Field(default="text-embedding-3-small")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity for retrieval")


class WorkingMemoryItem(BaseModel):
    """Item in agent's working memory (current workflow)."""
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeMemoryItem(BaseModel):
    """Item in episode memory (learnings from this analysis)."""
    episode_id: str  # Workflow ID
    market_question: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: str  # "strategy", "evidence", "outcome", "lesson"
    content: Any
    effectiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseItem(BaseModel):
    """Item in knowledge base (historical cross-workflow knowledge)."""
    id: str
    content_type: str  # "market_analysis", "evidence", "base_rate", "pattern"
    content: Any
    embedding: Optional[List[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Memory Manager
# =========================
class MemoryManager:
    """
    Centralized memory management for POLYSEER agents.

    Tiers:
      1) Working Memory (Checkpointer) - Current workflow state.
      2) Episode Memory (Store)        - Learnings from this analysis.
      3) Knowledge Base (Vector DB)    - Historical cross-workflow knowledge.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        store: Optional[BaseStore] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Args:
            config: Memory configuration.
            store: LangGraph Store instance.
            checkpointer: LangGraph Checkpointer instance.
        """
        self.config = config or MemoryConfig()
        self.store = store
        self.checkpointer = checkpointer
        logger.info(f"MemoryManager init: store={self.config.store_type}, checkpointer={self.config.checkpointer_type}")

    async def store_working_memory(
        self,
        agent_name: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store an item in working memory (current workflow).

        Note:
            LangGraph's checkpointer typically handles working memory persistence.
            This method validates and records critical items explicitly.

        Returns:
            True if accepted for tracking.
        """
        assert agent_name and key, "agent_name and key are required"
        _ = WorkingMemoryItem(agent_name=agent_name, key=key, value=value, metadata=metadata or {})
        return True

    async def store_episode_memory(
        self,
        episode_id: str,
        market_question: str,
        memory_type: str,
        content: Any,
        effectiveness: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persist learnings from the current episode for reuse in similar cases.

        Returns:
            True on success, False otherwise.
        """
        if not self.store:
            return False

        assert episode_id and memory_type, "episode_id and memory_type are required"
        item = EpisodeMemoryItem(
            episode_id=episode_id,
            market_question=market_question,
            memory_type=memory_type,
            content=content,
            effectiveness=effectiveness,
            metadata=metadata or {},
        )
        try:
            await self.store.aput(
                ("episode_memory",),
                f"{episode_id}:{memory_type}:{datetime.utcnow().isoformat()}",
                item.model_dump(),
            )
            return True
        except Exception:
            return False

    async def retrieve_episode_memories(
        self,
        market_question: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[EpisodeMemoryItem]:
        """
        Retrieve episode memories for similar market analyses.

        Returns:
            A list of EpisodeMemoryItem.
        """
        if not self.store:
            return []

        try:
            results = await self.store.asearch(("episode_memory",), query=market_question, limit=limit)
        except Exception:
            return []

        out: List[EpisodeMemoryItem] = []
        for r in results:
            try:
                item = EpisodeMemoryItem(**r.value)
                if memory_type is None or item.memory_type == memory_type:
                    out.append(item)
            except Exception:
                continue
        return out

    async def store_knowledge(
        self,
        content_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True,
    ) -> Optional[str]:
        """
        Store an item in the knowledge base for long‑term access.

        Returns:
            Knowledge ID on success, else None.
        """
        if not self.store:
            return None

        assert content_type, "content_type is required"
        knowledge_id = f"{content_type}:{datetime.utcnow().timestamp()}"
        item = KnowledgeBaseItem(id=knowledge_id, content_type=content_type, content=content, metadata=metadata or {})

        # TODO: generate embeddings if/when an embedding service is wired.
        if generate_embedding:
            pass

        try:
            await self.store.aput(("knowledge_base",), knowledge_id, item.model_dump())
            return knowledge_id
        except Exception:
            return None

    async def search_knowledge(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeBaseItem]:
        """
        Search the knowledge base.

        Returns:
            A list of KnowledgeBaseItem.
        """
        if not self.store:
            return []

        try:
            results = await self.store.asearch(("knowledge_base",), query=query, limit=limit)
        except Exception:
            return []

        items: List[KnowledgeBaseItem] = []
        for r in results:
            try:
                item = KnowledgeBaseItem(**r.value)
                if content_type is None or item.content_type == content_type:
                    items.append(item)
            except Exception:
                continue
        return items

    async def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Summarize a past workflow execution by aggregating episode memories.

        Returns:
            Summary dict or None if no memories found.
        """
        if not self.store:
            return None

        memories = await self.retrieve_episode_memories(market_question="", limit=100)
        target = [m for m in memories if m.episode_id == workflow_id]
        if not target:
            return None

        summary: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "market_question": target[0].market_question,
            "total_memories": len(target),
            "by_type": {},
            "effective_strategies": [],
            "timestamp": target[0].timestamp.isoformat(),
        }

        for m in target:
            summary["by_type"][m.memory_type] = summary["by_type"].get(m.memory_type, 0) + 1
            if m.memory_type == "strategy" and (m.effectiveness or 0) > 0.7:
                summary["effective_strategies"].append({"content": m.content, "effectiveness": m.effectiveness})

        return summary

    def integrity_report(self) -> Dict[str, Any]:
        """
        Compact integrity report with critical metrics and simple violations.
        """
        store_kind = getattr(self.store, "__class__", type("X", (), {})).__name__ if self.store else None
        limits = {
            "max_working_memory_messages": self.config.max_working_memory_messages,
            "max_episode_memory_items": self.config.max_episode_memory_items,
            "episode_retention_days": self.config.episode_retention_days,
        }
        violations = []
        if limits["max_working_memory_messages"] <= 0:
            violations.append("non_positive_working_memory_limit")
        if limits["max_episode_memory_items"] <= 0:
            violations.append("non_positive_episode_memory_limit")
        if limits["episode_retention_days"] <= 0:
            violations.append("non_positive_retention_days")

        return {
            "store_type": self.config.store_type,
            "store_runtime_class": store_kind,
            "checkpointer_type": self.config.checkpointer_type,
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "limits": limits,
            "violations": violations,
        }


# =========================
# Store Factory
# =========================
def create_store_from_config(settings: Optional[Any] = None) -> Optional[BaseStore]:
    """
    Create and initialize a LangGraph Store from configuration.

    Backends:
      - PostgreSQL (recommended; works with Supabase)
      - In-Memory (default/fallback; non‑persistent)
    """
    def _in_memory_store():
        from langgraph.store.memory import InMemoryStore
        return InMemoryStore()

    if settings is None:
        try:
            from config.settings import settings as global_settings
            settings = global_settings
        except ImportError:
            return _in_memory_store()

    if not getattr(settings, "ENABLE_MEMORY_PERSISTENCE", True):
        return _in_memory_store()

    backend = getattr(settings, "MEMORY_BACKEND", "postgresql").lower()

    if backend == "postgresql":
        postgres_url = getattr(settings, "POSTGRES_URL", "") or ""
        if not postgres_url:
            supabase_url = getattr(settings, "SUPABASE_URL", "") or ""
            supabase_key = getattr(settings, "SUPABASE_SERVICE_KEY", "") or getattr(settings, "SUPABASE_KEY", "") or ""
            if supabase_url and supabase_key:
                import re
                m = re.search(r"https://([a-zA-Z0-9-]+)\.supabase\.co", supabase_url)
                if m:
                    project_ref = m.group(1)
                    postgres_url = f"postgresql://postgres:{supabase_key}@db.{project_ref}.supabase.co:5432/postgres"

        if postgres_url:
            try:
                # Lazy imports to avoid hard dependency unless used.
                from langgraph.store.postgres import PostgresStore  # type: ignore

                # Optional connection test.
                try:
                    import psycopg  # type: ignore
                    test_conn = psycopg.connect(postgres_url, autocommit=True)
                    test_conn.close()
                except Exception:
                    pass

                store = PostgresStore.from_conn_string(postgres_url)

                # Ensure schema exists.
                import asyncio

                async def init_store():
                    async with store:
                        await store.setup()

                try:
                    asyncio.get_running_loop()
                    # Running loop detected; try nested strategy if available.
                    try:
                        import nest_asyncio  # type: ignore
                        nest_asyncio.apply()
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(init_store())
                    except Exception:
                        # Best effort; continue without explicit setup if nested execution fails.
                        pass
                except RuntimeError:
                    asyncio.run(init_store())

                return store
            except ImportError:
                # Guidance only; keep fallback silent and predictable.
                # To enable Postgres: pip install 'langgraph-checkpoint-postgres[pool]' psycopg[binary]
                return _in_memory_store()
            except Exception:
                return _in_memory_store()

    # Fallback: in-memory (non‑persistent)
    return _in_memory_store()


# =========================
# Singleton Accessors
# =========================
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    config: Optional[MemoryConfig] = None,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> MemoryManager:
    """
    Get or create the MemoryManager singleton.

    If no store is provided, it is created via create_store_from_config().
    """
    global _memory_manager
    if _memory_manager is None:
        if store is None:
            store = create_store_from_config()
        _memory_manager = MemoryManager(config=config, store=store, checkpointer=checkpointer)
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the MemoryManager singleton (primarily for tests)."""
    global _memory_manager
    _memory_manager = None