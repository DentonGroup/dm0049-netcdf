#!/usr/bin/env python3

"""Every acquisition file this project has ever written still loads.

resource/nc holds one real file per generation of the container, exactly as
the client of the day wrote it. The test is the whole round trip: upgrade the
file, then read it with the current reader. Nothing here asserts what the
upgrade does internally, so the container is free to keep changing; what is
pinned is that a file from any generation ends up readable.

Adding a generation is adding a file. It is a real acquisition, kept because a
file written by a version nobody runs any more is the only thing that can show
the upgrade still handles it.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dm0049_client.acquisition_file import read_base

from dm0049_netcdf import ACQUIRED_KEY
from dm0049_netcdf import SPECTRUM_KEY
from dm0049_netcdf import backup_path
from dm0049_netcdf import needs_upgrade
from dm0049_netcdf import upgrade_file

RESOURCE = Path(__file__).parent / "resource" / "nc"

# The committed files and their content, so a fixture cannot change without
# saying so, and a test that upgrades one in place instead of a copy of it is
# caught by the run that follows.
FIXTURES: dict[str, str] = {
    "1787475512.422__0001.nc": "302e01c4ba8e11e097b011d3d2a7959b0d31a122",
}


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def staged(name: str, tmp_path: Path) -> Path:
    """A writable copy keeping the name: the stem records when it was taken."""
    target = tmp_path / name
    shutil.copyfile(RESOURCE / name, target)
    return target


def test_every_committed_file_is_the_one_that_was_committed() -> None:
    """The fixtures are the pinning, so they are checked before anything else."""
    assert sorted(p.name for p in RESOURCE.glob("*.nc")) == sorted(FIXTURES)
    for name, digest in FIXTURES.items():
        assert sha1(RESOURCE / name) == digest, name


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_an_upgraded_file_loads(name: str, tmp_path: Path) -> None:
    """The whole point: a file from an older generation reads afterwards."""
    path = staged(name, tmp_path)
    upgrade_file(path)

    config = read_base(path)

    assert config[ACQUIRED_KEY] > 0
    assert not needs_upgrade(path)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_upgrading_keeps_the_original(name: str, tmp_path: Path) -> None:
    """Nothing is overwritten in place, so a bad upgrade is recoverable."""
    path = staged(name, tmp_path)
    before = path.read_bytes()

    original = upgrade_file(path, stamp=1787000000)

    assert original == backup_path(path, 1787000000)
    assert original.read_bytes() == before
    assert path.read_bytes() != before


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_upgrading_loses_nothing(name: str, tmp_path: Path) -> None:
    """Only the keys that were missing are added.

    An upgrade that quietly dropped a variable or re-typed one would satisfy
    every other test here, and the archive would carry the loss with nothing
    recording it.
    """
    path = staged(name, tmp_path)
    with xr.open_dataset(path) as dataset:
        before = {v: dataset[v].values.copy() for v in dataset.data_vars}
        attrs = dict(dataset.attrs)

    upgrade_file(path)

    with xr.open_dataset(path) as dataset:
        after = {v: dataset[v].values.copy() for v in dataset.data_vars}
        assert dict(dataset.attrs) == attrs
    assert not set(before) - set(after)
    for variable, value in before.items():
        # a real acquisition carries nan where a measurement did not apply, so
        # the comparison has to hold nan against nan
        np.testing.assert_array_equal(after[variable], value, err_msg=variable)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_upgrading_twice_does_nothing_the_second_time(
    name: str,
    tmp_path: Path,
) -> None:
    """Running over an archive again has to be a walk, not a second rewrite."""
    path = staged(name, tmp_path)
    upgrade_file(path)
    once = path.read_bytes()
    backups = sorted(tmp_path.glob("*.backup_*"))

    assert upgrade_file(path) is None
    assert path.read_bytes() == once
    assert sorted(tmp_path.glob("*.backup_*")) == backups


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_spectrum_survives(name: str, tmp_path: Path) -> None:
    """A file that carried one still does; the viewer needs it to draw."""
    path = staged(name, tmp_path)
    with xr.open_dataset(path) as dataset:
        if SPECTRUM_KEY not in dataset.data_vars:
            pytest.skip(f"{name} predates the spectrum being recorded")
        shape = dataset[SPECTRUM_KEY].shape

    upgrade_file(path)

    with xr.open_dataset(path) as dataset:
        assert dataset[SPECTRUM_KEY].shape == shape
