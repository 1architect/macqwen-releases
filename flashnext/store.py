"""Random-access reader for safetensors shards.

The checkpoint is larger than memory, so nothing is loaded whole. Each shard
file is memory-mapped once. Callers ask for the rows they need and get back a
small mx.array. Pages stay clean and file-backed, so macOS can drop them
without writing to swap.
"""
from __future__ import annotations

import ctypes
import json
import fcntl
import mmap
import os
import struct
import threading
from typing import Dict, Sequence, Tuple

import mlx.core as mx
import numpy as np

# safetensors dtype -> (numpy read dtype, mlx target dtype)
# BF16 has no numpy equivalent, so it travels as uint16 and is reinterpreted.
_DTYPES: Dict[str, Tuple[np.dtype, object]] = {
    "F32": (np.dtype("<f4"), mx.float32),
    "F16": (np.dtype("<f2"), mx.float16),
    "BF16": (np.dtype("<u2"), mx.bfloat16),
    "U32": (np.dtype("<u4"), mx.uint32),
    "I32": (np.dtype("<i4"), mx.int32),
    "I64": (np.dtype("<i8"), mx.int64),
    "U64": (np.dtype("<u8"), mx.uint64),
    "F64": (np.dtype("<f8"), mx.float32),
    "BOOL": (np.dtype("?"), mx.bool_),
    "U16": (np.dtype("<u2"), mx.uint16),
    "U8": (np.dtype("u1"), mx.uint8),
    "I8": (np.dtype("i1"), mx.int8),
}


class TensorRef:
    __slots__ = ("shard", "dtype", "shape", "start", "row_bytes")

    def __init__(self, shard: str, dtype: str, shape: Sequence[int], start: int):
        self.shard = shard
        self.dtype = dtype
        self.shape = tuple(shape)
        self.start = start
        trailing = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
        self.row_bytes = trailing * _DTYPES[dtype][0].itemsize


class SafeTensorStore:
    """Memory-mapped view over every shard of a safetensors checkpoint."""

    def __init__(self, model_dir: str):
        self.dir = os.path.expanduser(model_dir)
        self.refs: Dict[str, TensorRef] = {}
        self._maps: Dict[str, mmap.mmap] = {}
        self._files: Dict[str, object] = {}
        self._fds: Dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._views: Dict[str, np.ndarray] = {}
        self._shared_views: Dict[str, np.ndarray] = {}
        self._drop_ngram = os.environ.get("FLASHNEXT_NGRAM_DONTNEED") == "1"
        self._read_mode = os.environ.get("FLASHNEXT_READ", "pread")
        self._pread_chunk = int(os.environ.get("FLASHNEXT_PREAD_CHUNK", "1"))
        self._hybrid_cutoff = int(os.environ.get("FLASHNEXT_HYBRID_CUTOFF", "2"))
        self._sort_reads = os.environ.get("FLASHNEXT_SORT_READS") == "1"
        self._no_cache = os.environ.get("FLASHNEXT_F_NOCACHE") == "1"
        self._dlpack = os.environ.get("FLASHNEXT_DLPACK", "1") == "1"
        self._mmap_advice = os.environ.get("FLASHNEXT_MMAP_ADVICE", "random")
        self._pinned = []

        index_path = os.path.join(self.dir, "model.safetensors.index.json")
        with open(index_path) as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))

        for shard in shards:
            self.add_shard(shard)

    def add_shard(self, shard: str) -> None:
        """Register an extra safetensors shard beside the indexed checkpoint."""
        path = os.path.join(self.dir, shard)
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            self.refs[name] = TensorRef(
                shard,
                meta["dtype"],
                meta["shape"],
                data_start + meta["data_offsets"][0],
            )

    def _view(self, name: str) -> np.ndarray:
        """A numpy view of one tensor, mapped once and reused.

        Fancy-indexing this view gathers rows in C. Reading them one at a time
        through np.frombuffer costs a Python round trip per row, which measured
        250 MB/s against an SSD that does 3 GB/s.
        """
        view = self._views.get(name)
        if view is None:
            ref = self.refs[name]
            read_dtype, _ = _DTYPES[ref.dtype]
            view = np.memmap(
                os.path.join(self.dir, ref.shard),
                dtype=read_dtype,
                mode="r",
                offset=ref.start,
                shape=ref.shape,
            )
            # Without this the kernel reads ahead around every scattered row.
            # The advice on self._maps never reached here: np.memmap opens its
            # own mapping, so the two are unrelated.
            self._advise_view(view)
            self._views[name] = view
        return view

    def _advise_view(self, view: np.ndarray) -> None:
        advice = {
            "normal": mmap.MADV_NORMAL,
            "random": mmap.MADV_RANDOM,
            "sequential": mmap.MADV_SEQUENTIAL,
        }[self._mmap_advice]
        try:
            view._mmap.madvise(advice)
        except (AttributeError, OSError):
            pass

    def set_mmap_advice(self, value: str) -> None:
        self._mmap_advice = value
        for view in self._views.values():
            self._advise_view(view)
        advice = {
            "normal": mmap.MADV_NORMAL,
            "random": mmap.MADV_RANDOM,
            "sequential": mmap.MADV_SEQUENTIAL,
        }[value]
        for handle in self._maps.values():
            try:
                handle.madvise(advice)
            except (AttributeError, OSError):
                pass

    def _map(self, shard: str) -> mmap.mmap:
        handle = self._maps.get(shard)
        if handle is None:
            file = open(os.path.join(self.dir, shard), "rb")
            handle = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
            advice = {
                "normal": mmap.MADV_NORMAL,
                "random": mmap.MADV_RANDOM,
                "sequential": mmap.MADV_SEQUENTIAL,
            }[self._mmap_advice]
            handle.madvise(advice)
            self._files[shard] = file
            self._maps[shard] = handle
        return handle

    def _shared_view(self, name: str) -> np.ndarray:
        view = self._shared_views.get(name)
        if view is None:
            ref = self.refs[name]
            read_dtype, _ = _DTYPES[ref.dtype]
            view = np.ndarray(
                ref.shape,
                dtype=read_dtype,
                buffer=self._map(ref.shard),
                offset=ref.start,
            )
            self._shared_views[name] = view
        return view

    def shape(self, name: str) -> Tuple[int, ...]:
        return self.refs[name].shape

    def rows_np(
        self, name: str, indices: Sequence[int], read_mode: str | None = None
    ) -> np.ndarray:
        """Gather rows as numpy. Safe to call from worker threads: the page
        faults happen inside numpy, which drops the GIL, so concurrent calls
        keep the NVMe queue busy instead of idling between serial reads."""
        rows = indices if isinstance(indices, list) else list(indices)
        mode = read_mode or self._read_mode
        if mode == "hybrid":
            mode = "shared_mmap" if len(rows) <= self._hybrid_cutoff else "pread"
        if mode in ("pread", "preadv"):
            return self._rows_pread(name, rows, mode)
        view = (
            self._shared_view(name)
            if mode == "shared_mmap"
            else self._view(name)
        )
        out = np.ascontiguousarray(view[rows])
        if self._drop_ngram and ".ngram_embedding." in name:
            try:
                view._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        return out

    def pin_rows(self, name: str, rows: Sequence[int]) -> int:
        """Keep selected file-backed rows resident without copying them."""
        ref = self.refs[name]
        view = self._shared_view(name)
        base = int(view.__array_interface__["data"][0])
        page = mmap.PAGESIZE
        libc = ctypes.CDLL(None, use_errno=True)
        total = 0
        for row in rows:
            address = base + int(row) * ref.row_bytes
            start = address - address % page
            end = (address + ref.row_bytes + page - 1) // page * page
            length = end - start
            if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(length)) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            self._pinned.append((start, length))
            total += length
        return total

    def unpin_all(self) -> None:
        libc = ctypes.CDLL(None)
        for address, length in self._pinned:
            libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(length))
        self._pinned.clear()

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
                    if self._no_cache:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    self._fds[shard] = fd
        return fd

    def _rows_pread(
        self, name: str, rows: Sequence[int], read_mode: str
    ) -> np.ndarray:
        """Read expert rows with explicit positioned I/O instead of mmap faults."""
        ref = self.refs[name]
        dtype, _ = _DTYPES[ref.dtype]
        out = np.empty((len(rows), *ref.shape[1:]), dtype=dtype)
        fd = self._fd(ref.shard)
        for slot, row in enumerate(rows):
            offset = ref.start + row * ref.row_bytes
            if read_mode == "preadv":
                read = os.preadv(fd, [memoryview(out[slot]).cast("B")], offset)
            else:
                data = os.pread(fd, ref.row_bytes, offset)
                read = len(data)
                if read == ref.row_bytes:
                    out[slot] = np.frombuffer(data, dtype=dtype).reshape(ref.shape[1:])
            if read != ref.row_bytes:
                raise OSError(f"short pread for {name} row {row}")
        return out

    def to_mx(self, name: str, out: np.ndarray) -> mx.array:
        """Wrap a numpy gather as mx. Call from the main thread only."""
        _, target = _DTYPES[self.refs[name].dtype]
        array = mx.from_dlpack(out, copy=False) if self._dlpack else mx.array(out)
        if target is mx.bfloat16:
            array = array.view(mx.bfloat16)
        return array

    def rows(self, name: str, indices: Sequence[int]) -> mx.array:
        """Return the requested rows of `name`, stacked along axis 0."""
        ref = self.refs[name]
        _, target = _DTYPES[ref.dtype]
        out = self.rows_np(name, indices)

        array = mx.array(out)
        # uint16 -> bfloat16 is a reinterpret, not a conversion.
        if target is mx.bfloat16:
            array = array.view(mx.bfloat16)
        return array

    def whole_np(self, name: str) -> np.ndarray:
        """The full tensor as numpy. One sequential pass, no gather."""
        return np.asarray(self._view(name))

    def whole(self, name: str) -> mx.array:
        return self.rows(name, range(self.refs[name].shape[0]))

    def close(self) -> None:
        self.unpin_all()
        self._shared_views.clear()
        for handle in self._maps.values():
            handle.close()
        for file in self._files.values():
            file.close()
        for fd in self._fds.values():
            os.close(fd)
        self._views.clear()
        self._maps.clear()
        self._files.clear()
