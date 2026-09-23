"""Scale and bias rows stored contiguously per expert (research, off).

A cold Q4/G32 expert costs nine positioned reads: three 819,200-byte weight
rows and six 102,400-byte scale and bias rows, each in a different tensor. The
drive serves this pattern at about 1.9 GB/s during decode whatever the worker
count (2026-09-23). This sidecar stores an expert's six small rows as one
614,400-byte record, in the order the expert-major stream record lays them out
(gate scales, gate biases, up scales, up biases, down scales, down biases), so
one ``preadv`` scatters it into the record's three scale-and-bias gaps. A cold
expert then costs four reads. The bytes are copies of the checkpoint's own
rows; arithmetic does not change.

``FLASHNEXT_SMALL_SIDECAR`` names the sidecar directory. Only layers listed in
its manifest use it, and only on the stream-pack path. The manifest records
the checkpoint identity; a mismatch disables the sidecar.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from .slab_pack import (
    DOWN_SCALES_OFFSET,
    GATE_SCALES_OFFSET,
    RECORD_STRIDE,
    UP_SCALES_OFFSET,
)
from .store import _profiled_preadv

FLAG = "FLASHNEXT_SMALL_SIDECAR"
FORMAT = "flashnext-small-sidecar-v1"
PARTS = (
    ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj", "scales"), ("up_proj", "biases"),
    ("down_proj", "scales"), ("down_proj", "biases"),
)
ROW_BYTES = 102_400
PAIR_BYTES = 2 * ROW_BYTES
RECORD_BYTES = 6 * ROW_BYTES
# Where each scale-and-bias pair lands inside one stream record.
_PAIR_TARGETS = (GATE_SCALES_OFFSET, UP_SCALES_OFFSET, DOWN_SCALES_OFFSET)


def layer_path(directory, layer: int) -> Path:
    return Path(directory) / f"layer-{int(layer):02d}.bin"


class SmallSidecar:
    """Open read descriptors for the manifest's layers."""

    def __init__(self, directory, layers):
        self.directory = Path(directory)
        self.layers = frozenset(int(layer) for layer in layers)
        self._fds = {}

    def fd(self, layer: int) -> int:
        fd = self._fds.get(layer)
        if fd is None:
            fd = os.open(layer_path(self.directory, layer), os.O_RDONLY)
            # Match the checkpoint descriptors: no kernel read-ahead.
            try:
                fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
            except OSError:
                pass
            self._fds[layer] = fd
        return fd

    def read_into(self, layer: int, experts, buffer, first_slot: int) -> None:
        """Scatter each expert's record into its stream record, one read each."""
        fd = self.fd(layer)
        view = memoryview(buffer)
        for index, expert in enumerate(experts):
            base = (first_slot + index) * RECORD_STRIDE
            targets = [
                view[base + offset : base + offset + PAIR_BYTES]
                for offset in _PAIR_TARGETS
            ]
            read = _profiled_preadv(fd, targets, int(expert) * RECORD_BYTES)
            if read != RECORD_BYTES:
                raise OSError(f"short sidecar read, layer {layer} expert {expert}")

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def for_store(store) -> SmallSidecar | None:
    """The sidecar named by the environment, if it matches this checkpoint."""
    cached = getattr(store, "_flashnext_small_sidecar", False)
    if cached is not False:
        return cached
    sidecar = None
    directory = os.environ.get(FLAG, "")
    if directory:
        directory = Path(os.path.expanduser(directory))
        manifest = json.loads((directory / "manifest.json").read_text())
        from .routing import checkpoint_identity_for_store

        if manifest.get("format") != FORMAT:
            raise ValueError(f"unknown small sidecar format in {directory}")
        if manifest.get("checkpoint_identity") != checkpoint_identity_for_store(store):
            raise ValueError(f"small sidecar {directory} belongs to another checkpoint")
        sidecar = SmallSidecar(directory, manifest["layers"])
    try:
        store._flashnext_small_sidecar = sidecar
    except AttributeError:
        pass
    return sidecar


def build(store, layers, directory, prefix_for_layer) -> dict:
    """Write one sidecar file per layer, bypassing the page cache.

    ``prefix_for_layer(layer)`` returns the switch_mlp tensor prefix. Every
    record is read back and compared with the checkpoint rows.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source_fd(shard):
        fd = sources.get(shard)
        if fd is None:
            fd = os.open(os.path.join(store.dir, shard), os.O_RDONLY)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            sources[shard] = fd
        return fd

    def original_record(prefix, expert):
        pieces = []
        for projection, part in PARTS:
            ref = store.refs[f"{prefix}.{projection}.{part}"]
            if ref.row_bytes != ROW_BYTES:
                raise ValueError(f"{prefix}.{projection}.{part}: row is {ref.row_bytes} bytes")
            pieces.append(os.pread(
                source_fd(ref.shard), ROW_BYTES, ref.start + expert * ref.row_bytes,
            ))
        return b"".join(pieces)

    written = {}
    try:
        for layer in layers:
            prefix = prefix_for_layer(layer)
            experts = int(store.shape(f"{prefix}.gate_proj.scales")[0])
            path = layer_path(directory, layer)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                for expert in range(experts):
                    record = original_record(prefix, expert)
                    if os.pwrite(fd, record, expert * RECORD_BYTES) != RECORD_BYTES:
                        raise OSError(f"short sidecar write {path}")
                os.fsync(fd)
            finally:
                os.close(fd)
            check = os.open(path, os.O_RDONLY)
            try:
                fcntl.fcntl(check, fcntl.F_NOCACHE, 1)
                for expert in range(experts):
                    if os.pread(check, RECORD_BYTES, expert * RECORD_BYTES) != (
                        original_record(prefix, expert)
                    ):
                        raise ValueError(f"sidecar record differs: layer {layer} expert {expert}")
            finally:
                os.close(check)
            written[layer] = experts
    finally:
        for fd in sources.values():
            os.close(fd)
    return written
