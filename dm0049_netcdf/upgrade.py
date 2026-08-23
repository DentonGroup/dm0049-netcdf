#!/usr/bin/env python3

"""Bringing an acquisition file up to the current container.

An acquisition written before the container named a key does not carry it, and
nothing downstream can invent it per read. The upgrade adds what is derivable
and refuses what is not, rather than leaving an archive in a state where some
files answer a question and others do not for reasons nobody recorded.

The original is never overwritten in place: it is renamed beside itself and the
new file written at the old path, so a failure part way leaves the original
under a name that says what it is.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr

from .access import run as netcdf_run
from .container import ACQUIRED_KEY

# cycloidal-client's own name for the same instant, written before the
# container had one
LEGACY_ACQUIRED_KEY: str = "current_spectrum_timestamp"

# an epoch second in 2001 and in 2100; a stem outside this is not a timestamp
EPOCH_FLOOR: float = 1_000_000_000.0
EPOCH_CEILING: float = 4_100_000_000.0


class CannotUpgrade(ValueError):
    """A file the upgrade has no derivable answer for."""


def _stem_timestamp(path: Path) -> float | None:
    """The epoch second a capture's name starts with, if it is one.

    Files are named <epoch>__<n>.nc by the client that took them, so the name
    is a record of when even in a file that recorded nothing else. It is the
    weaker source of the two: a rename destroys it, which is why the upgrade
    exists at all.
    """
    head = path.stem.split("__", 1)[0]
    if not head.replace(".", "", 1).isdigit():
        return None
    value = float(head)
    if not EPOCH_FLOOR <= value <= EPOCH_CEILING:
        return None
    return value


def acquired_for(dataset: xr.Dataset, path: Path) -> float:
    """When the acquisition in dataset was taken.

    Preferring what the file recorded over what its name says, because the
    name is the thing that does not survive being moved.
    """
    for source in (dataset.attrs, dataset.data_vars):
        if LEGACY_ACQUIRED_KEY in source:
            value = source[LEGACY_ACQUIRED_KEY]
            if isinstance(value, xr.DataArray):
                value = value.values.reshape(-1)[0]
            return float(value)
    stamp = _stem_timestamp(path)
    if stamp is None:
        raise CannotUpgrade(
            f"{path.as_posix()}: no {LEGACY_ACQUIRED_KEY} and the name does "
            f"not begin with an epoch second, so when it was taken is not "
            f"recorded anywhere in it"
        )
    return stamp


def needs_upgrade(path: Path) -> bool:
    """Whether path is missing anything the current container names."""

    def read() -> bool:
        with xr.open_dataset(path) as dataset:
            return (
                ACQUIRED_KEY not in dataset.attrs
                and ACQUIRED_KEY not in dataset.data_vars
            )

    return netcdf_run("needs_upgrade", path, read)


def backup_path(path: Path, stamp: float | None = None) -> Path:
    """Where the original goes. Named so it cannot be mistaken for a capture."""
    when = int(time.time() if stamp is None else stamp)
    return path.with_name(f"{path.name}.backup_{when}")


def upgrade_file(path: Path, *, stamp: float | None = None) -> Path | None:
    """Add what the current container names. Returns where the original went.

    Returns None for a file that already carries everything, so a second run
    over an archive is a walk and nothing else.
    """
    path = Path(path)
    if not needs_upgrade(path):
        return None

    def read() -> xr.Dataset:
        with xr.open_dataset(path) as dataset:
            return dataset.load()

    dataset = netcdf_run("upgrade_read", path, read)
    acquired = acquired_for(dataset, path)
    dataset[ACQUIRED_KEY] = xr.DataArray(
        np.asarray([acquired], dtype=np.float64), dims=["index"]
    )

    original = backup_path(path, stamp)
    if original.exists():
        raise CannotUpgrade(
            f"{original.as_posix()} already exists; the original would be lost"
        )
    path.rename(original)

    def write() -> None:
        dataset.to_netcdf(path)

    netcdf_run("upgrade_write", path, write)
    return original
