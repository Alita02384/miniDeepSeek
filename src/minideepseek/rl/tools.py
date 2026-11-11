"""Minimal tool registry and executor for multi-turn agent RL.

Refactored from the MiniMind ``trainer/train_agent.py`` reference
implementation (Apache-2.0).  The tool surface is intentionally tiny and
deterministic so GRPO/OPD rollouts are reproducible without external services:

* ``calculate_math``, ``unit_converter``, ``get_current_weather``,
  ``get_current_time``, ``get_exchange_rate``, ``translate_text``.

The original mock tables are kept so the MiniMind ``agent_rl*.jsonl`` records
remain usable without changing their ``gt`` expectations.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Callable
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048}

_TEMPERATURE_ALIASES = {"c": "celsius", "f": "fahrenheit", "k": "kelvin"}


def convert_unit_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value, using affine formulas for the temperature scales.

    The old implementation multiplied by 1.8 / 0.5556, which ignored the
    offset (10 C -> 18 instead of 50; 18 F -> 10.0008 instead of -7.7778).
    """
    source = _TEMPERATURE_ALIASES.get(str(from_unit).lower(), str(from_unit).lower())
    target = _TEMPERATURE_ALIASES.get(str(to_unit).lower(), str(to_unit).lower())
    if source == target:
        return float(value)
    if source == "celsius" and target == "fahrenheit":
        return float(value) * 9.0 / 5.0 + 32.0
    if source == "fahrenheit" and target == "celsius":
        return (float(value) - 32.0) * 5.0 / 9.0
    if source == "celsius" and target == "kelvin":
        return float(value) + 273.15
    if source == "kelvin" and target == "celsius":
        return float(value) - 273.15
    if source == "fahrenheit" and target == "kelvin":
        return (float(value) - 32.0) * 5.0 / 9.0 + 273.15
    if source == "kelvin" and target == "fahrenheit":
        return (float(value) - 273.15) * 9.0 / 5.0 + 32.0
    factor = UNIT_DATA.get(f"{source}_{target}")
    return float(value) * float(factor) if factor is not None else float(value)


def safe_math_eval(expression: str) -> Any:
    """Evaluate arithmetic with an AST allow-list (no ``eval``)."""
    def pow_guard(base: float, exp: float) -> float:
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100):
            raise ValueError("power too large")
        return base**exp

    consts = {"pi": math.pi, "e": math.e, "tau": math.tau}
    funcs = {
        "pow": pow_guard,
        **{
            name: getattr(math, name)
            for name in [
                "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan",
                "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
                "floor", "ceil", "trunc", "fabs", "fmod", "hypot", "gcd",
                "degrees", "radians",
            ]
        },
    }
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: pow_guard,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def resolve(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", "") == "math":
            return node.attr
        return ""

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        name = resolve(node)
        if name in consts:
            return consts[name]
        if isinstance(node, ast.UnaryOp):
            return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp):
            return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords:
            return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    chars = str.maketrans({"^": "**", "×": "*", "÷": "/", "−": "-", "²": "**2", "³": "**3", "（": "(", "）": ")"})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512:
        raise ValueError("expression empty or too long")
    return walk(ast.parse(expr, mode="eval").body)


MOCK_RESULTS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "unit_converter": lambda args: {"result": round(convert_unit_value(float(args.get("value", 0)), str(args.get("from_unit", "")), str(args.get("to_unit", ""))), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

CHECK_ARGS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict):
            calls.append(call)
    return calls


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    fn = MOCK_RESULTS.get(name)
    if fn is None:
        return None
    try:
        return fn(args)
    except Exception:
        return None


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def valid_tool_call(call: dict[str, Any], tools: list[dict[str, Any]] | None) -> bool:
    name = str(call.get("name", ""))
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False
    if not isinstance(args, dict):
        return False
    check = CHECK_ARGS.get(name)
    return bool(name in tool_names(tools) and check and check(args))


def normalize_tool_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}
