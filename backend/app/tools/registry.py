# -*- coding: utf-8 -*-
"""自研工具注册表（L3 工具层 · "一切皆插件"）。

对应开发前文件 §3.6.3 工具调用 + §3.6.4 借鉴 DSH"一切皆插件"：
把 ASR / 姿态 / OCR / 时间轴切片 / RAG 检索 / 记忆读写 统一封装为**可插拔工具**，
配置化注册到工具注册表，Agent 通过工具名发现并调度。热卸一个工具不影响其他，
替换实现（如本地 ASR → 云端 ASR）只改注册，不改 Agent 逻辑。

设计：
- ToolSpec：工具的元数据（name、description、参数 schema、可选的输出 schema）。
- registry：全局工具注册表，支持 register / get / list / call / unregister。
- 每个工具 = 一个普通 Python 函数 + 一份 ToolSpec，函数即实现、即插件。
- 预留 MCP 适配：ToolSpec 与 MCP tool schema 对齐，阶段4 可无痛桥接到 MCP。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable

# 可观测钩子（L6）：工具调用时回调，用于轨迹日志。默认空实现，由外层注入。
_on_call_hooks: list[Callable[[str, str, dict], None]] = []


def add_on_call_hook(hook: Callable[[str, str, dict], None]) -> None:
    """注册一个工具调用观察钩子（hook(name, category, kwargs)）。"""
    _on_call_hooks.append(hook)


class ToolError(Exception):
    """工具调用失败。"""


@dataclass
class ToolSpec:
    """工具元数据（对齐 MCP tool 结构，便于后续协议化）。"""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)   # JSON Schema
    output_schema: dict[str, Any] | None = None                # 可选，输出结构描述
    category: str = "general"                                  # rag/audio/vision/board/memory/...
    enabled: bool = True


class ToolRegistry:
    """可插拔工具注册表。"""

    def __init__(self, name: str = "default"):
        self.name = name
        self._tools: dict[str, tuple[Callable, ToolSpec]] = {}

    # ---------- 注册 / 发现 ----------
    def register(self, spec: ToolSpec, fn: Callable) -> None:
        self._tools[spec.name] = (fn, spec)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_spec(self, name: str) -> ToolSpec | None:
        t = self._tools.get(name)
        return t[1] if t else None

    def list(self, category: str | None = None) -> list[dict]:
        out = []
        for name, (_, spec) in self._tools.items():
            if category and spec.category != category:
                continue
            out.append({"name": name, "description": spec.description,
                        "category": spec.category, "enabled": spec.enabled})
        return out

    def call(self, name: str, **kwargs: Any) -> Any:
        """调度工具：解析 → 调用 → 可观测日志。name 缺失/不可用则抛 ToolError。"""
        item = self._tools.get(name)
        if not item:
            raise ToolError(f"工具 '{name}' 未注册")
        fn, spec = item
        if not spec.enabled:
            raise ToolError(f"工具 '{name}' 已停用")
        # 可观测：回调所有钩子（轨迹日志）
        for hook in _on_call_hooks:
            try:
                hook(name, spec.category, kwargs)
            except Exception:
                pass
        return fn(**kwargs)


# 全局默认注册表
registry = ToolRegistry("agent_tools")
