"""Runtime tools (`python_exec`, `web_search`, `file_tool`, `unit_tester`)."""

from __future__ import annotations

from flybrain.runtime.tools.base import Tool, ToolRegistry, ToolResult
from flybrain.runtime.tools.file_tool import FileTool
from flybrain.runtime.tools.python_exec import PythonExecTool
from flybrain.runtime.tools.unit_tester import UnitTesterTool
from flybrain.runtime.tools.web_search import WebSearchTool
from flybrain.runtime.tools.yandex_search import YandexSearchConfig, YandexSearchTool


def default_tool_registry(*, live_search: bool = False) -> ToolRegistry:
    """Registry with deterministic-by-default tools wired in.

    Pass ``live_search=True`` to swap the fixture-based
    ``WebSearchTool`` for a real :class:`YandexSearchTool` that hits
    the Yandex Search API. The live retriever requires
    ``YANDEX_FOLDER_ID`` + ``YANDEX_SEARCH_API_KEY`` (or
    ``YANDEX_API_KEY``) in the environment; the swap fails fast
    otherwise.
    """
    reg = ToolRegistry()
    reg.register(PythonExecTool())
    reg.register(FileTool())
    reg.register(YandexSearchTool() if live_search else WebSearchTool())
    reg.register(UnitTesterTool())
    return reg


__all__ = [
    "FileTool",
    "PythonExecTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "UnitTesterTool",
    "WebSearchTool",
    "YandexSearchConfig",
    "YandexSearchTool",
    "default_tool_registry",
]
