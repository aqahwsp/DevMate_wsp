from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SOURCE_FILES = sorted(
    (Path(__file__).resolve().parents[1] / "src" / "devmate").glob("*.py")
)
BANNED_DEFAULT_CALLS = {
    "Body",
    "File",
    "Form",
    "Query",
    "typer.Option",
}
E203_SLICE_PATTERN = re.compile(r"\[[^\]]*\s:\s[^\]]*\]")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_every_source_file_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_no_line_exceeds_88_characters(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    too_long = [
        (index, len(line), line)
        for index, line in enumerate(lines, start=1)
        if len(line) > 88
    ]
    assert not too_long, f"{path.name} has lines longer than 88 chars: {too_long}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_no_e203_whitespace_before_slice_colon(path: Path) -> None:
    offenders = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if E203_SLICE_PATTERN.search(line):
            offenders.append((index, line))
    assert not offenders, f"{path.name} contains E203-like slice spacing: {offenders}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_no_b008_style_default_calls(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = list(node.args.defaults)
        defaults.extend(item for item in node.args.kw_defaults if item is not None)
        for default in defaults:
            if not isinstance(default, ast.Call):
                continue
            name = _call_name(default.func)
            if name in BANNED_DEFAULT_CALLS:
                offenders.append((node.name, default.lineno, name))
    assert not offenders, f"{path.name} contains B008-style defaults: {offenders}"
