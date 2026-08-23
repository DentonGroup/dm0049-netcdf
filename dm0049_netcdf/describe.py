#!/usr/bin/env python3

"""Rendering an acquisition file for a person to read.

An acquisition carries a few hundred scalars, a handful of arrays with a
thousand elements each, and JSON-string attributes holding nested structures.
ncdump prints the schema and then every value; a bare xarray repr prints the
schema and no values. What is wanted between them is every name, with enough
of each value to recognise it and nothing like all of a spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from .access import run as netcdf_run

# elements of an array shown before the rest is summarised
ARRAY_HEAD = 6
# characters of a string shown before it is cut
STRING_WIDTH = 96
# nesting depth of a decoded JSON attribute before it is summarised
JSON_DEPTH = 4

INDENT = "  "


def _fits(text: str) -> str:
    """A single line, cut to width, with the cut made visible."""
    text = text.replace("\n", "\\n")
    if len(text) <= STRING_WIDTH:
        return text
    return f"{text[:STRING_WIDTH]}… ({len(text)} chars)"


def _number(value) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.6g}"
    return str(value)


def _element(value) -> str:
    """One element of an array. A string element is cut like any other."""
    if isinstance(value, bytes):
        return _fits(value.decode("utf-8", errors="replace"))
    if isinstance(value, (str, np.str_)):
        return _fits(str(value))
    return _number(value)


def _array(values: np.ndarray) -> str:
    """The head of an array, then what was left out.

    A spectrum is 1704 numbers and none of them is recognisable on its own;
    the first few and the range say what it is.
    """
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return "[]"
    head = ", ".join(_element(v) for v in flat[:ARRAY_HEAD])
    if flat.size <= ARRAY_HEAD:
        return f"[{head}]"
    tail = f"… {flat.size - ARRAY_HEAD} more"
    if np.issubdtype(flat.dtype, np.number):
        finite = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.floating) else flat
        if finite.size:
            tail += f", min {_number(finite.min())}, max {_number(finite.max())}"
    return f"[{head}, {tail}]"


def _scalar(value) -> str:
    if isinstance(value, bytes):
        return _fits(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _fits(value)
    if isinstance(value, np.ndarray):
        return _array(value) if value.ndim else _number(value.item())
    return _number(value)


def _json_lines(value, depth: int = 0) -> list[str]:
    """A decoded JSON attribute, one line per leaf, summarised past a depth."""
    pad = INDENT * depth
    if depth >= JSON_DEPTH:
        return [f"{pad}… nested {type(value).__name__}"]
    if isinstance(value, dict):
        if not value:
            return [f"{pad}{{}}"]
        lines = []
        for key in value:
            child = value[key]
            if isinstance(child, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_json_lines(child, depth + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(child)}")
        return lines
    if isinstance(value, list):
        if len(value) > ARRAY_HEAD and not any(
            isinstance(v, (dict, list)) for v in value
        ):
            return [f"{pad}{_array(np.asarray(value))}"]
        lines = []
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                lines.append(f"{pad}[{index}]:")
                lines.extend(_json_lines(child, depth + 1))
            else:
                lines.append(f"{pad}[{index}] {_scalar(child)}")
        return lines
    return [f"{pad}{_scalar(value)}"]


def _decoded(text: str):
    """A JSON attribute decoded, or None when it is not one."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    # attributes carrying JSON are written by build_dataset; one that will not
    # decode is an ordinary string that happens to start with a brace
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def render(path: Path) -> list[str]:
    """Every name in an acquisition file, with a readable part of each value."""
    path = Path(path)
    lines = [
        f"{path.as_posix()}",
        f"{INDENT}{path.stat().st_size:,} bytes",
    ]

    def read() -> list[str]:
        out: list[str] = []
        with xr.open_dataset(path) as ds:
            out.append("")
            out.append("dimensions")
            for name, size in ds.sizes.items():
                out.append(f"{INDENT}{name}: {size}")

            out.append("")
            out.append(f"attributes ({len(ds.attrs)})")
            for name in sorted(ds.attrs):
                value = ds.attrs[name]
                decoded = _decoded(value) if isinstance(value, str) else None
                if decoded is None:
                    out.append(f"{INDENT}{name}: {_scalar(value)}")
                else:
                    out.append(f"{INDENT}{name}:")
                    out.extend(
                        f"{INDENT}{line}" for line in _json_lines(decoded, depth=1)
                    )

            out.append("")
            out.append(f"coordinates ({len(ds.coords)})")
            for name in ds.coords:
                var = ds.coords[name]
                out.append(
                    f"{INDENT}{name} {tuple(var.dims)} {var.dtype} "
                    f"{_array(var.values)}"
                )

            out.append("")
            out.append(f"variables ({len(ds.data_vars)})")
            width = max((len(str(n)) for n in ds.data_vars), default=0)
            for name in sorted(ds.data_vars):
                var = ds.data_vars[name]
                shape = "×".join(str(s) for s in var.shape) or "scalar"
                out.append(
                    f"{INDENT}{str(name):<{width}}  {shape:>12}  {str(var.dtype):>8}  "
                    f"{_array(var.values)}"
                )
        return out

    lines.extend(netcdf_run("nc_inspect", path, read))
    return lines
