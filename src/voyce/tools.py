from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .events import ToolCallEvent


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Coroutine[Any, Any, str]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCallEvent) -> str:
        tool = self.get_tool(tool_call.name)
        if not tool:
            return f"Error: Tool '{tool_call.name}' not found."
        try:
            # Arguments are passed as keyword arguments
            result = await tool.func(**tool_call.arguments)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_call.name}': {e}"
