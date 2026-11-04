"""JSON Schema -> regex (the subset that maps cleanly to a regular language).

Supported: ``type`` string / integer / number / boolean / null / array / object, ``enum``,
``const``, ``anyOf`` / ``oneOf``, string ``minLength`` / ``maxLength`` / ``pattern``, array
``items`` with
``minItems``/``maxItems``, nested objects. Object properties are emitted in schema order and all
listed properties are produced (required or not), which keeps the language regular and the output
predictable. Whitespace: compact JSON with an optional single space after ``:`` and ``,``.
"""

from __future__ import annotations

import json
import re

WS = "[ ]?"
STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
INTEGER = r"-?(?:0|[1-9][0-9]{0,15})"
NUMBER = INTEGER + r"(?:\.[0-9]{1,15})?(?:[eE][+-]?[0-9]{1,3})?"

_META = set(r".^$*+?()[]{}|\/")


def _escape(text: str) -> str:
    return "".join("\\" + c if c in _META else c for c in text)


def _literal(value) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def schema_to_regex(schema: dict | bool, max_unbounded_items: int | None = None) -> str:
    if schema is True or schema == {}:
        raise ValueError("unconstrained schema ({}): use a concrete type")
    if "const" in schema:
        return _literal(schema["const"])
    if "enum" in schema:
        return "(?:" + "|".join(_literal(v) for v in schema["enum"]) + ")"
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return "(?:" + "|".join(schema_to_regex(s) for s in schema[key]) + ")"
    t = schema.get("type")
    if isinstance(t, list):
        return "(?:" + "|".join(schema_to_regex({**schema, "type": x}) for x in t) + ")"
    if t == "string":
        if "pattern" in schema:
            pat = schema["pattern"].removeprefix("^").removesuffix("$")
            return '"' + pat + '"'
        lo, hi = schema.get("minLength", 0), schema.get("maxLength")
        rep = f"{{{lo},{'' if hi is None else hi}}}"
        return '"' + STRING_CHAR + rep + '"'
    if t == "integer":
        return INTEGER
    if t == "number":
        return NUMBER
    if t == "boolean":
        return "(?:true|false)"
    if t == "null":
        return "null"
    if t == "array":
        item = schema_to_regex(schema.get("items", {"type": "string"}))
        lo, hi = schema.get("minItems", 0), schema.get("maxItems", max_unbounded_items)
        if hi == 0:
            return r"\[\]"
        more_lo = max(lo - 1, 0)
        more_hi = "" if hi is None else hi - 1
        body = f"{item}(?:,{WS}{item}){{{more_lo},{more_hi}}}"
        return rf"\[(?:{body})\]" if lo > 0 else rf"\[(?:{body})?\]"
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if not props:
            raise ValueError("objects need explicit properties")
        parts = [f'"{_escape(k)}":{WS}{schema_to_regex(v)}' for k, v in props.items()]
        return r"\{" + WS + f",{WS}".join(parts) + WS + r"\}"
    raise ValueError(f"unsupported schema: {schema}")


def choice_regex(choices: list[str]) -> str:
    return "(?:" + "|".join(_escape(c) for c in choices) + ")"


def validate(instance, schema) -> bool:
    """Tiny validator for the same subset (used in tests and the demo)."""
    if "const" in schema:
        return instance == schema["const"]
    if "enum" in schema:
        return instance in schema["enum"]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return any(validate(instance, s) for s in schema[key])
    t = schema.get("type", "object" if "properties" in schema else None)
    if isinstance(t, list):
        return any(validate(instance, {**schema, "type": x}) for x in t)
    if t == "string":
        ok = isinstance(instance, str)
        if ok and "pattern" in schema:
            ok = (
                re.fullmatch(schema["pattern"].removeprefix("^").removesuffix("$"), instance)
                is not None
            )
        return ok and schema.get("minLength", 0) <= len(instance) <= schema.get(
            "maxLength", 1 << 30
        )
    if t == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if t == "number":
        return isinstance(instance, int | float) and not isinstance(instance, bool)
    if t == "boolean":
        return isinstance(instance, bool)
    if t == "null":
        return instance is None
    if t == "array":
        return (
            isinstance(instance, list)
            and schema.get("minItems", 0) <= len(instance) <= schema.get("maxItems", 1 << 30)
            and all(validate(x, schema.get("items", {"type": "string"})) for x in instance)
        )
    if t == "object":
        props = schema.get("properties", {})
        return (
            isinstance(instance, dict)
            and set(instance) == set(props)
            and all(validate(instance[k], props[k]) for k in props)
        )
    return False
