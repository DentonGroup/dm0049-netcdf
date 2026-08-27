#!/usr/bin/env python3

"""An acquisition file as a table.

The container holds four shapes of thing and a CSV holds one, so the export
has to say which of them go in the table and which go above it.

    (index, frame, pixel)   spectrum_raw, the frames themselves
    (index, pixel)          spectrum, the reduced value, when the file has one
    (index, frame)          frame_times
    (index,) and attributes everything else: the gain, the clocks, the build

The first three go in the table, one row per pixel per frame per acquisition,
with the lower rank ones repeated down it. That repetition is the price of a
flat table and it is what makes each row stand alone, which is the only reason
to want a CSV rather than the file it came from.

The scalars and the attributes go above it as comment lines. They are constant
down a column and putting them in one would add thirty columns of the same
value repeated a quarter of a million times, and every reader worth using
takes a comment character.

Only spectrum_raw and frame_times are required. `spectrum` is written by some
acquisitions and not others, and an array on any other dimension -- a settle
profile, say -- shares no axis with a row of this table and cannot be a column
of it. Both cases are named in the header, because a file that came out
narrower than another should say so on its own face rather than in whatever
read it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .access import run as netcdf_run
from .container import SPECTRUM_KEY

COMMENT: str = "#"

# What the table carries, in column order. Anything else in the file is a
# scalar or an attribute and goes in the header.
REQUIRED: tuple[str, ...] = (SPECTRUM_KEY, "frame_times")
POSITION: tuple[str, ...] = ("index", "frame", "pixel", "frame_time_s")
OPTIONAL: tuple[str, ...] = ("spectrum",)


def columns_for(dataset: xr.Dataset) -> tuple[str, ...]:
    """The table's columns for this file, which depend on what it carries."""
    present = tuple(name for name in OPTIONAL if name in dataset.data_vars)
    return POSITION + (SPECTRUM_KEY,) + present

# Enough digits that a float64 ADC code survives the round trip, and few
# enough that an integral one is still written as an integer.
NUMBER_FORMAT: str = "%.10g"


class MissingArray(KeyError):
    """The file does not carry an array the table is built from."""


def _untabulated(dataset: xr.Dataset) -> list[str]:
    """Arrays the table has no axis for, so a reader knows they were left."""
    tabulated = {*REQUIRED, *OPTIONAL}
    return sorted(
        f"{name} {tuple(dataset[name].dims)}"
        for name in dataset.data_vars
        if name not in tabulated and dataset[name].dims not in ((), ("index",))
    )


def _header_lines(dataset: xr.Dataset, path: Path) -> list[str]:
    """The scalars and attributes, one per line, above the table."""
    lines = [
        f"{COMMENT} source: {path.as_posix()}",
        f"{COMMENT} dimensions: "
        + ", ".join(f"{name}={size}" for name, size in dataset.sizes.items()),
    ]
    for name in sorted(dataset.attrs):
        value = str(dataset.attrs[name]).replace("\n", "\\n")
        lines.append(f"{COMMENT} attr {name}: {value}")
    for name in sorted(dataset.data_vars):
        variable = dataset.data_vars[name]
        if variable.dims != ("index",):
            continue
        values = ", ".join(
            str(v).replace("\n", "\\n") for v in np.asarray(variable.values).ravel()
        )
        lines.append(f"{COMMENT} {name}: {values}")
    left = _untabulated(dataset)
    if left:
        lines.append(
            f"{COMMENT} not tabulated, no axis in common with a row: "
            + ", ".join(left)
        )
    lines.append(f"{COMMENT} columns: " + ",".join(columns_for(dataset)))
    return lines


def _table(dataset: xr.Dataset) -> np.ndarray:
    """One row per pixel per frame per acquisition, in COLUMNS order."""
    missing = [name for name in REQUIRED if name not in dataset.data_vars]
    if missing:
        raise MissingArray(
            f"the file carries no {', '.join(missing)}; a table of frames "
            f"cannot be built from an acquisition that did not record them"
        )

    frames = np.asarray(dataset[SPECTRUM_KEY].values, dtype=np.float64)
    times = np.asarray(dataset["frame_times"].values, dtype=np.float64)
    acquisitions, frame_count, pixels = frames.shape

    stack = [
        np.repeat(np.arange(acquisitions), frame_count * pixels),
        np.tile(np.repeat(np.arange(frame_count), pixels), acquisitions),
        np.tile(np.arange(pixels), acquisitions * frame_count),
        np.repeat(times.ravel(), pixels),
        frames.ravel(),
    ]
    for name in OPTIONAL:
        if name in dataset.data_vars:
            values = np.asarray(dataset[name].values, dtype=np.float64)
            stack.append(np.repeat(values, frame_count, axis=0).ravel())
    return np.column_stack(stack)


def to_csv(path: Path | str, destination: Path | str | None = None) -> Path:
    """Write an acquisition file out as a CSV. Returns what it wrote.

    The destination defaults to the acquisition's own name with a .csv
    suffix, and an existing file is an error rather than something to write
    over: a capture and its exports share a stem, so the name that is already
    there was almost certainly made from the same file and losing it silently
    would be worse than stopping.
    """
    source = Path(path)
    target = Path(destination) if destination is not None else source.with_suffix(".csv")
    if target.exists():
        raise FileExistsError(
            f"{target.as_posix()} is already there; remove it or name another "
            f"destination"
        )

    def read() -> tuple[list[str], tuple[str, ...], np.ndarray]:
        with xr.open_dataset(source) as dataset:
            return (
                _header_lines(dataset, source),
                columns_for(dataset),
                _table(dataset),
            )

    header, columns, table = netcdf_run("nc_to_csv", source, read)

    with target.open("w", encoding="utf8") as handle:
        for line in header:
            handle.write(f"{line}\n")
        handle.write(",".join(columns) + "\n")
        np.savetxt(handle, table, delimiter=",", fmt=NUMBER_FORMAT)
    return target
