#!/usr/bin/env python3

"""The netCDF container every acquisition is written into.

One convention, whichever client writes it: a single "index" dimension (size 1
per file, files stack along it on the analysis side), run-constant metadata as
dataset attributes, per-acquisition scalars as (index,) data variables, nested
values flattened into dotted keys, and selected structured values carried as
JSON-string attributes.

This module owns the mechanics and the arrays. The keys are the writer's: a
client hands its own registries in, and the container never changes for them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pint
import pint_xarray  # noqa: F401 – registers pint accessor on xarray
import xarray as xr

from .access import run as netcdf_run


FORMAT_VERSION = 1
FORMAT_VERSION_KEY = "format_version"
# .nc files written before the key was renamed carry the original misspelling.
# Reading it is the one accommodation the format makes for older files; nothing
# writes it.
LEGACY_FORMAT_VERSION_KEY = "jason_file_version"

# the array that makes an acquisition displayable, as opposed to the ones
# derived from it
# When the acquisition was taken, epoch seconds. Every acquisition has one and
# every client knows it, so the container names it rather than each client
# naming it differently. A file's own name is not a record of this: it does not
# survive being renamed, and a copy carries the copy's mtime.
ACQUIRED_KEY: str = "acquired"

# Nested values flattened into dotted data variables. "ext" is the container's
# own name for a caller-owned tree; a writer with more of them passes its own
# set, as dm0049-client does for the settle window.
CONTAINER_FLATTEN_KEYS: frozenset[str] = frozenset({"ext"})

SPECTRUM_KEY: str = "spectrum_raw"
ARRAY_KEYS: tuple[str, ...] = (SPECTRUM_KEY, "spectrum", "frame_times")
SEED_KEYS: frozenset[str] = frozenset({"pixels", *ARRAY_KEYS})

# raw ADC counts, matching the analysis side
ureg = pint.UnitRegistry(force_ndarray_like=True)
ureg.define("ADC = count")


def spectrum_dataset(
    *,
    spectrum_raw: np.ndarray,
    spectrum: np.ndarray,
    frame_times: np.ndarray,
    pixels: np.ndarray,
) -> xr.Dataset:
    """The array half of an acquisition, shaped for the container.

    spectrum_raw is (frame, pixel), the orientation the reduction produces
    transposed once here rather than at each call site. Every array gains a
    leading index axis of one so files stack along index on the analysis side.

    This is the format, so it lives with the format. A client that adds its
    own scalars still gets its arrays from here, which is what makes a file
    dm0049-client wrote and a file a client above it wrote the same file.
    """
    spectrum_raw = np.asarray(spectrum_raw, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.int32)

    if spectrum_raw.ndim != 2:
        raise ValueError(f"spectrum_raw must be (frame, pixel), got {spectrum_raw.shape}")
    frame_count, pixel_count = spectrum_raw.shape
    if pixel_count != pixels.size:
        raise ValueError(
            f"spectrum_raw has {pixel_count} pixels, pixels has {pixels.size}"
        )
    if frame_times.size != frame_count:
        raise ValueError(
            f"spectrum_raw has {frame_count} frames, frame_times has "
            f"{frame_times.size}"
        )

    ds = xr.Dataset(
        {
            "spectrum_raw": xr.DataArray(
                spectrum_raw[np.newaxis, ...],
                dims=["index", "frame", "pixel"],
                attrs={"long_name": "Raw spectrum frames", "units": "ADC"},
            ),
            "spectrum": xr.DataArray(
                spectrum[np.newaxis, ...],
                dims=["index", "pixel"],
                attrs={"long_name": "Averaged spectrum", "units": "ADC"},
            ),
            "frame_times": xr.DataArray(
                frame_times[np.newaxis, ...],
                dims=["index", "frame"],
                attrs={"long_name": "Frame acquisition times", "units": "s"},
            ),
        },
        coords={
            "index": [0],
            "pixel": pixels,
            "frame": np.arange(frame_count, dtype=np.int32),
        },
    )
    return ds.pint.quantify(
        {"spectrum_raw": "ADC", "spectrum": "ADC", "frame_times": "s"},
        unit_registry=ureg,
    )

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dicts into dotted keys, matching pandas.json_normalize.

    e.g. {"ext": {"pump_data": {"pressure": 5}}} -> {"ext.pump_data.pressure": 5}
    """
    items: dict = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def unflatten_dict(flat: dict, sep: str = ".") -> dict:
    """Inverse of flatten_dict."""
    out: dict = {}
    for key, value in flat.items():
        *parents, leaf = key.split(sep)
        node = out
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return out


def _scalar(value):
    """netCDF hands back numpy scalars; a config dict holds python ones."""
    return value.item() if isinstance(value, np.generic) else value


def build_dataset(
    config_dict: dict,
    *,
    scalar_keys: tuple[str, ...],
    constant_keys: frozenset[str],
    json_attr_keys: frozenset[str] = frozenset(),
    flatten_keys: frozenset[str] = CONTAINER_FLATTEN_KEYS,
    seed: xr.Dataset | None = None,
    seed_keys: frozenset[str] = frozenset(),
) -> xr.Dataset:
    """Place a config dict into a dataset following the container convention.

    seed: a dataset already carrying the caller's arrays and coordinates
    (spectra, pixels, ...); the base contributes scalars, attributes, ext, and
    JSON attributes on top of it. seed_keys names the config keys the seed
    consumed, so validation knows they are placed.

    A key with no placement is an error, not a silent loss.
    """
    unplaced = (
        set(config_dict)
        - set(scalar_keys)
        - json_attr_keys
        - flatten_keys
        - seed_keys
    )
    if unplaced:
        raise ValueError(
            f"no netCDF placement for {sorted(unplaced)}: add them to "
            f"the scalar keys or they are silently lost on write"
        )

    ds = seed if seed is not None else xr.Dataset(coords={"index": [0]})

    for key in scalar_keys:
        if key not in config_dict:
            continue
        if key in constant_keys:
            ds.attrs[key] = config_dict[key]
        else:
            ds[key] = xr.DataArray(np.asarray([config_dict[key]]), dims=["index"])

    for key in sorted(json_attr_keys):
        if key in config_dict:
            ds.attrs[key] = json.dumps(config_dict[key])

    for root in sorted(flatten_keys):
        if config_dict.get(root) is None:
            continue
        # Flatten into dotted keys (ext.pump_data.pressure, settle_window.tau)
        # matching the json_normalize naming used on the analysis side. Every
        # leaf becomes a data variable carrying a leading "index" dimension
        # (size 1 for this single acquisition) so files stack cleanly along
        # index. Scalars are (index,) shaped; sequence leaves get a trailing
        # "<key>_dim" dimension (e.g. ext.gas_delivery_data.channels_dim),
        # which fixup_xarray maps to the "channel" dimension.
        for key, value in flatten_dict({root: config_dict[root]}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)[np.newaxis, ...]  # prepend index axis
                ds[key] = xr.DataArray(arr, dims=["index", f"{key}_dim"])
            else:
                ds[key] = xr.DataArray(np.asarray([value]), dims=["index"])

    return ds


def write_acquisition(ds: xr.Dataset, data_file: Path, *, compress: bool) -> None:
    """Write one acquisition dataset to data_file."""
    ds = ds.pint.dequantify()
    encoding = None
    if compress:
        # zlib (DEFLATE) compression. complevel 4 is a good speed/size
        # trade-off; higher levels cost much more CPU for marginal size gains.
        # Shrinks the float64 spectral arrays well. Only numeric, at least 1-D
        # variables can be chunked/compressed -- scalars and string/vlen vars
        # (e.g. "experiment") raise a netCDF filter error.
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name, var in ds.data_vars.items()
            if var.ndim >= 1 and np.issubdtype(var.dtype, np.number)
        }

    def write() -> None:
        ds.to_netcdf(data_file, encoding=encoding)

    netcdf_run("write_acquisition", data_file, write)


def nested_from_flat(config: dict, root: str) -> dict | None:
    """Reassemble one flattened tree out of a config dict.

    build_dataset writes a nested value as dotted keys and dataset_to_dict
    reads them back as it finds them, because the analysis side names ext
    leaves that way. A caller that wants back the tree it wrote asks here
    rather than for a key that is no longer in the file.

    Returns None when nothing under the root is present, which is what a file
    written without it looks like.
    """
    prefix = f"{root}."
    leaves = {
        key[len(prefix) :]: value
        for key, value in config.items()
        if key.startswith(prefix)
    }
    if not leaves:
        return None
    nested: dict = {}
    for dotted, value in leaves.items():
        parts = dotted.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def dataset_to_dict(
    ds: xr.Dataset,
    *,
    array_keys: tuple[str, ...] = (),
    json_attr_keys: frozenset[str] = frozenset(),
) -> dict:
    """The config dict an acquisition was written from.

    array_keys names the caller's per-acquisition arrays; ones the file does
    not carry are simply absent from the result, so a base file (metadata
    only) reads with the same code as a full acquisition.
    """
    index_size = ds.sizes["index"]
    if index_size != 1:
        raise ValueError(
            f"expected one acquisition, got {index_size} along index. Stacked "
            f"files are an analysis-side artifact; the client displays one"
        )
    row = ds.isel(index=0)

    config = {key: _scalar(value) for key, value in ds.attrs.items()}
    for key in json_attr_keys:
        if key in config:
            config[key] = json.loads(config[key])

    if "pixel" in ds.coords:
        config["pixels"] = ds.coords["pixel"].values.tolist()
    for name in array_keys:
        if name in row:
            config[name] = row[name].values.tolist()

    ext_flat: dict = {}
    for name, var in row.data_vars.items():
        if name in array_keys:
            continue
        # (index,) scalars are 0-d after isel; ext list leaves keep their dim
        value = var.values.tolist() if var.ndim else var.values.item()
        if name.startswith("ext."):
            ext_flat[name] = value
        else:
            config[name] = value
    config.update(unflatten_dict(ext_flat))

    # after the data variables, so a file that recorded the version as a
    # variable rather than an attribute is normalized too
    if LEGACY_FORMAT_VERSION_KEY in config:
        config[FORMAT_VERSION_KEY] = config.pop(LEGACY_FORMAT_VERSION_KEY)

    return config
