#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")
MAX_EXPANSION_ROUNDS = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a JSON object.")
    return data


def _get_by_dot_path(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(path)
        cur = cur[key]
    return cur


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class ConfigResolver:
    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._path_cache: dict[str, Any] = {}
        self._resolving_paths: set[str] = set()

    def resolve_all(self) -> dict[str, Any]:
        return self._resolve_value(self._data, "<root>")

    def resolve_path(self, path: str) -> Any:
        if path in self._path_cache:
            return self._path_cache[path]
        if path in self._resolving_paths:
            chain = " -> ".join(list(self._resolving_paths) + [path])
            raise ValueError(f"Cyclic config placeholder reference detected: {chain}")

        raw = _get_by_dot_path(self._data, path)
        self._resolving_paths.add(path)
        try:
            resolved = self._resolve_value(raw, path)
        finally:
            self._resolving_paths.remove(path)
        self._path_cache[path] = resolved
        return resolved

    def _resolve_value(self, value: Any, current_path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: self._resolve_value(
                    child,
                    key if current_path == "<root>" else f"{current_path}.{key}",
                )
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._resolve_value(child, f"{current_path}[{idx}]")
                for idx, child in enumerate(value)
            ]
        if isinstance(value, str):
            return self._resolve_string(value, current_path)
        return value

    def _resolve_string(self, value: str, current_path: str) -> str:
        if "${" not in value:
            return value

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            resolved = self._resolve_token(token, current_path)
            if isinstance(resolved, (dict, list)):
                raise ValueError(
                    f"Placeholder '{token}' in '{current_path}' resolved to a non-scalar value."
                )
            return _stringify_scalar(resolved)

        out = value
        for _ in range(MAX_EXPANSION_ROUNDS):
            new_out = PLACEHOLDER_RE.sub(_replace, out)
            if new_out == out:
                return new_out
            out = new_out
        raise ValueError(
            f"Too many placeholder expansion rounds while resolving '{current_path}'. "
            "Check for recursive references."
        )

    def _resolve_token(self, token: str, current_path: str) -> Any:
        if token.startswith(("env:", "ENV:")):
            env_name = token.split(":", 1)[1]
            if env_name not in os.environ:
                raise KeyError(
                    f"Missing environment variable '{env_name}' referenced by '{current_path}'."
                )
            return os.environ[env_name]

        try:
            return self.resolve_path(token)
        except KeyError:
            if token in os.environ:
                return os.environ[token]
            raise KeyError(
                f"Unknown placeholder '{token}' referenced by '{current_path}'."
            ) from None


def _flatten_items(data: dict[str, Any], prefix: str = ""):
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_items(value, full_key)
        elif value is None:
            continue
        elif isinstance(value, bool):
            yield full_key, ("true" if value else "false")
        else:
            yield full_key, str(value)


def _cmd_flatten(config_file: Path) -> int:
    resolver = ConfigResolver(_load_json(config_file))
    resolved = resolver.resolve_all()
    for key, value in _flatten_items(resolved):
        print(f"{key}={value}")
    return 0


def _cmd_get(config_file: Path, path: str) -> int:
    resolver = ConfigResolver(_load_json(config_file))
    try:
        value = resolver.resolve_path(path)
    except KeyError:
        return 0

    if isinstance(value, (dict, list)):
        print(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
    elif isinstance(value, bool):
        print("true" if value else "false")
    elif value is not None:
        print(value)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve placeholders in co-learning JSON configs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    flatten_parser = subparsers.add_parser(
        "flatten",
        help="Resolve a config and print flattened key=value pairs.",
    )
    flatten_parser.add_argument("config", type=Path)

    get_parser = subparsers.add_parser(
        "get",
        help="Resolve a config path and print its value.",
    )
    get_parser.add_argument("config", type=Path)
    get_parser.add_argument("path", type=str)

    args = parser.parse_args()

    try:
        if args.command == "flatten":
            return _cmd_flatten(args.config)
        if args.command == "get":
            return _cmd_get(args.config, args.path)
    except Exception as exc:
        print(f"[config_resolver] {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
