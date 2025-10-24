import sys
import types


def _ensure_pydantic_settings_stub() -> None:
    """Provide a lightweight BaseSettings replacement for tests."""
    if "pydantic_settings" in sys.modules:
        return

    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.BaseSettings = _BaseSettings
    sys.modules["pydantic_settings"] = module


def _ensure_clob_client_stub() -> None:
    """Install lightweight stubs for optional external dependencies used in tools."""
    if "py_clob_client.client" in sys.modules:
        return

    clob_module = types.ModuleType("py_clob_client")
    client_module = types.ModuleType("py_clob_client.client")

    class _StubClobClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_orderbook(self, *args, **kwargs):
            return {}

    client_module.ClobClient = _StubClobClient
    clob_module.client = client_module

    sys.modules["py_clob_client"] = clob_module
    sys.modules["py_clob_client.client"] = client_module


def _ensure_valyu_stub() -> None:
    """Stub Valyu LangChain tools to avoid optional dependency imports."""
    if "langchain_valyu" in sys.modules:
        return

    module = types.ModuleType("langchain_valyu")

    class _StubTool:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, *args, **kwargs):
            raise NotImplementedError("Valyu tool is not available in test environment.")

    module.ValyuSearchTool = _StubTool
    module.ValyuContentsTool = _StubTool
    sys.modules["langchain_valyu"] = module


def _ensure_langchain_core_stubs() -> None:
    """Provide minimal LangChain core primitives for import-time usage."""
    if "langchain_core.messages" not in sys.modules:
        messages_module = types.ModuleType("langchain_core.messages")

        class _Message:
            def __init__(self, content: str = ""):
                self.content = content

        class HumanMessage(_Message):
            pass

        class SystemMessage(_Message):
            pass

        messages_module.HumanMessage = HumanMessage
        messages_module.SystemMessage = SystemMessage
        sys.modules["langchain_core.messages"] = messages_module

    if "langchain_core.tools" not in sys.modules:
        tools_module = types.ModuleType("langchain_core.tools")

        class BaseTool:
            pass

        def tool(func):
            return func

        tools_module.tool = tool
        tools_module.BaseTool = BaseTool
        sys.modules["langchain_core.tools"] = tools_module


def _ensure_langchain_openai_stub() -> None:
    """Stub ChatOpenAI for import-time references."""
    if "langchain_openai" in sys.modules:
        return

    module = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, *args, **kwargs):
            raise NotImplementedError("ChatOpenAI stub does not support execution in tests.")

    module.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = module


def _ensure_langgraph_stub() -> None:
    """Stub langgraph structures used at import time."""
    if "langgraph.store.base" not in sys.modules:
        base_module = types.ModuleType("langgraph.store.base")

        class SearchItem:
            pass

        class BaseStore:
            pass

        base_module.SearchItem = SearchItem
        base_module.BaseStore = BaseStore
        sys.modules["langgraph.store.base"] = base_module

    if "langgraph.checkpoint.base" not in sys.modules:
        checkpoint_module = types.ModuleType("langgraph.checkpoint.base")

        class BaseCheckpointSaver:
            pass

        checkpoint_module.BaseCheckpointSaver = BaseCheckpointSaver
        sys.modules["langgraph.checkpoint.base"] = checkpoint_module


_ensure_pydantic_settings_stub()
_ensure_clob_client_stub()
_ensure_valyu_stub()
_ensure_langchain_core_stubs()
_ensure_langchain_openai_stub()
_ensure_langgraph_stub()
