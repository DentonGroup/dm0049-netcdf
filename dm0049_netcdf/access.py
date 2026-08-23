#!/usr/bin/env python3

"""The one path into libnetcdf.

libnetcdf and the HDF5 beneath it keep global state and are not thread safe
unless they were built for it. xarray carries locks for this, and they are not
enough: two of its own paths close a netCDF file without holding them.

    CachingFileManager.__del__ calls close(needs_lock=False) at garbage
    collection time, on whichever thread happens to run the collection.

    FILE_CACHE is an LRU of 128 open files whose eviction callback closes the
    file it drops. Eviction runs under the cache's own lock, which is not the
    netCDF lock, on whichever thread opened the file that caused it.

So a reader walking a directory of more than 128 acquisitions closes a netCDF
file on its own thread, off the lock, once per open. Overlap that with a write
and libnetcdf loses track of its own dimensions:

    nc4hdf.c:932: var_create_dataset:
        Assertion `dim && dim->hdr.id == var->dimids[d]' failed

or corrupts the allocator, depending on where the timing lands.

Serialising our own calls cannot fix that, because the unsynchronised closes
are not our calls. What fixes it is that libnetcdf only ever sees one thread:
every operation runs here, on the gateway, so the collections and the evictions
run here too.

Callers keep their threads. A reader thread submits and blocks, which is the
same thing it was doing before, and the GUI thread is no more blocked than it
was.

None of this is gated. The trace runs on every operation, and entering
libnetcdf from any other thread is fatal rather than reported, because a
process that continues past this invariant is a process whose files are
already suspect.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")

# a C level abort carries no python traceback of its own, and the aborts this
# module exists to prevent are inside libnetcdf
faulthandler.enable(file=sys.stderr, all_threads=True)


class NetCDFThreadViolation(Exception):
    """A libnetcdf operation was attempted off the gateway thread."""


class NetCDFGatewayClosed(Exception):
    """The gateway is gone; the interpreter is on its way out."""


def _emit(text: str) -> None:
    # flushed: what this reports precedes a refused operation, and a buffer
    # that never drains reports nothing
    sys.stderr.write(f"{text}\n")
    sys.stderr.flush()


_GATEWAY = ThreadPoolExecutor(max_workers=1, thread_name_prefix="netcdf-gateway")
# eager, so the thread that owns libnetcdf is known before anything asks
_GATEWAY_THREAD: int = _GATEWAY.submit(threading.get_ident).result()


def on_gateway_thread() -> bool:
    return threading.get_ident() == _GATEWAY_THREAD


def run(kind: str, path: Path | str, work: Callable[[], T]) -> T:
    """Perform one libnetcdf operation on the gateway thread.

    work must leave nothing open and return nothing lazy: a dataset handed
    back to another thread is closed by whichever thread collects it, which is
    the fault this exists to remove.
    """
    if on_gateway_thread():
        return _guarded(kind, path, work)
    try:
        future = _GATEWAY.submit(_guarded, kind, path, work)
    except RuntimeError as e:
        # concurrent.futures shuts its executors down from an atexit hook, so
        # a thread still working while the interpreter exits gets a bare
        # RuntimeError about scheduling. Name it: a caller that outlives the
        # gateway needs to stop, not to report a scheduling problem.
        if "shutdown" not in str(e):
            raise
        raise NetCDFGatewayClosed(
            f"{kind} {path}: the interpreter is shutting down and the netcdf "
            f"gateway is closed; stop before it does"
        ) from e
    return future.result()


def _guarded(kind: str, path: Path | str, work: Callable[[], T]) -> T:
    with NetCDFOp(kind, path):
        return work()


class NetCDFOp:
    """The guard on one libnetcdf operation.

    Refuses to run off the gateway thread. That refusal is the whole point:
    an operation reaching libnetcdf from anywhere else is the fault this
    module exists to remove, and a process that continues past it is one whose
    files are already suspect.
    """

    def __init__(self, kind: str, path: Path | str) -> None:
        self.kind = kind
        self.path = str(path)
        self.thread = threading.get_ident()
        self.thread_name = threading.current_thread().name

    def __enter__(self) -> "NetCDFOp":
        if not on_gateway_thread():
            _emit(
                f"REFUSED {self.kind} {self.path}: libnetcdf may only be "
                f"entered from tid={_GATEWAY_THREAD}, this is tid="
                f"{self.thread} {self.thread_name}. Route it through "
                f"netcdf_access.run()."
            )
            traceback.print_stack(file=sys.stderr)
            sys.stderr.flush()
            raise NetCDFThreadViolation(
                f"{self.kind} {self.path} on tid={self.thread} "
                f"{self.thread_name}, gateway is tid={_GATEWAY_THREAD}"
            )

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
