"""Slab Pack: Pre-extracted, page-aligned, file-backed resident expert memory.

Layout:
  Page 0 (4096 bytes):
    - Magic (4 bytes): 0x4D4F4553 ("MOES")
    - Version (4 bytes): 1
    - Expert count (4 bytes): N
    - Record stride (4 bytes): 3,072,000 bytes (750 x 4096-byte pages)
    - Header size (4 bytes): 4096 bytes
    - Model identity hash (8 bytes)
    - Reserved (4 bytes)
    - Directory entries (N * 8 bytes):
        layer_id (uint16), expert_id (uint16), global_slot (uint32)

  Pages 1+ (N * 3,072,000 bytes):
    Each expert record is exactly 3,072,000 bytes (naturally 4K page-aligned):
      - gate_proj.weight: offset 0 (819,200 bytes, uint32)
      - gate_proj.scales: offset 819,200 (102,400 bytes, bfloat16)
      - gate_proj.biases: offset 921,600 (102,400 bytes, bfloat16)
      - up_proj.weight:   offset 1,024,000 (819,200 bytes, uint32)
      - up_proj.scales:   offset 1,843,200 (102,400 bytes, bfloat16)
      - up_proj.biases:   offset 1,945,600 (102,400 bytes, bfloat16)
      - down_proj.weight: offset 2,048,000 (819,200 bytes, uint32)
      - down_proj.scales: offset 2,867,200 (102,400 bytes, bfloat16)
      - down_proj.biases: offset 2,969,600 (102,400 bytes, bfloat16)
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

HEADER_MAGIC = 0x4D4F4553  # "MOES"
HEADER_VERSION = 1
HEADER_SIZE = 4096
RECORD_STRIDE = 3072000
DIRECTORY_OFFSET = 32
DIRECTORY_ENTRY_SIZE = 8
MAX_DIRECTORY_ENTRIES = (HEADER_SIZE - DIRECTORY_OFFSET) // DIRECTORY_ENTRY_SIZE

# Projections and sub-component offsets
GATE_WEIGHT_OFFSET = 0
GATE_SCALES_OFFSET = 819200
GATE_BIASES_OFFSET = 921600

UP_WEIGHT_OFFSET = 1024000
UP_SCALES_OFFSET = 1843200
UP_BIASES_OFFSET = 1945600

DOWN_WEIGHT_OFFSET = 2048000
DOWN_SCALES_OFFSET = 2867200
DOWN_BIASES_OFFSET = 2969600

_PARTS = ("weight", "scales", "biases")
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

_LIBC = None
_MLOCK = None
_MUNLOCK = None


def _init_libc():
    global _LIBC, _MLOCK, _MUNLOCK
    if _LIBC is not None:
        return
    try:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        _MLOCK = _LIBC.mlock
        _MLOCK.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _MLOCK.restype = ctypes.c_int

        _MUNLOCK = _LIBC.munlock
        _MUNLOCK.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _MUNLOCK.restype = ctypes.c_int
    except Exception:
        _MLOCK = None
        _MUNLOCK = None


def libc_mlock(ptr: int, size: int) -> bool:
    """Lock memory range into physical RAM."""
    _init_libc()
    if _MLOCK is None:
        return False
    try:
        return _MLOCK(ctypes.c_void_p(ptr), ctypes.c_size_t(size)) == 0
    except Exception:
        return False


def libc_munlock(ptr: int, size: int) -> bool:
    """Unlock memory range."""
    _init_libc()
    if _MUNLOCK is None:
        return False
    try:
        return _MUNLOCK(ctypes.c_void_p(ptr), ctypes.c_size_t(size)) == 0
    except Exception:
        return False


def build_slab_pack(
    store: Any,
    allocation: Mapping[int, Sequence[int]],
    output_path: str | Path,
    model_hash: bytes = b"\x00" * 8,
) -> int:
    """Extract expert weights from store and write a page-aligned slab pack.

    Returns the total file size in bytes.
    """
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp")

    # Flatten allocation in deterministic order: sorted by layer_id
    ordered_experts: List[Tuple[int, int, int]] = []
    global_slot = 0
    for layer_id in sorted(allocation.keys()):
        for expert_id in allocation[layer_id]:
            ordered_experts.append((layer_id, int(expert_id), global_slot))
            global_slot += 1

    expert_count = len(ordered_experts)
    if expert_count > MAX_DIRECTORY_ENTRIES:
        raise ValueError(
            f"Slab pack directory requires {expert_count} entries, "
            f"but the {HEADER_SIZE}-byte header supports only "
            f"{MAX_DIRECTORY_ENTRIES}"
        )
    total_size = HEADER_SIZE + expert_count * RECORD_STRIDE

    with open(temp_path, "wb") as f:
        # 1. Write Page 0 Header
        hdr = bytearray(HEADER_SIZE)
        struct.pack_into(
            "<IIIII8sI",
            hdr,
            0,
            HEADER_MAGIC,
            HEADER_VERSION,
            expert_count,
            RECORD_STRIDE,
            HEADER_SIZE,
            model_hash[:8].ljust(8, b"\x00"),
            0,  # reserved
        )

        # Directory table starting at offset 32: (layer_id: u16, expert_id: u16, slot: u32)
        dir_offset = DIRECTORY_OFFSET
        for layer_id, expert_id, slot in ordered_experts:
            struct.pack_into("<HHI", hdr, dir_offset, layer_id, expert_id, slot)
            dir_offset += 8

        f.write(hdr)

        # 2. Write Expert Records
        for layer_id, expert_id, slot in ordered_experts:
            prefix = f"language_model.model.layers.{layer_id}.mlp.switch_mlp"
            record_bytes = bytearray(RECORD_STRIDE)
            rec_offset = 0

            for proj in _PROJECTIONS:
                for part in _PARTS:
                    name = f"{prefix}.{proj}.{part}"
                    # Read single row as numpy array
                    row = store.rows_np(name, [expert_id])
                    row_bytes = row.tobytes()
                    record_bytes[rec_offset : rec_offset + len(row_bytes)] = row_bytes
                    rec_offset += len(row_bytes)

            if rec_offset != RECORD_STRIDE:
                raise ValueError(
                    f"Record size mismatch for layer {layer_id} expert {expert_id}: "
                    f"got {rec_offset} bytes, expected {RECORD_STRIDE}"
                )
            f.write(record_bytes)

    os.replace(temp_path, out_path)
    return total_size


class SlabPack:
    """Memory-mapped, mlocked slab pack providing a single zero-copy MTLBuffer."""

    __slots__ = (
        "path",
        "lock_memory",
        "fd",
        "_mm",
        "size",
        "expert_count",
        "layer_to_base_slot",
        "layer_expert_to_slot",
        "buffer_np",
        "buffer_mx",
        "is_locked",
    )

    def __init__(self, path: str | Path, lock_memory: bool = True):
        self.path = Path(path).expanduser()
        self.lock_memory = lock_memory
        self.fd: int | None = None
        self._mm: mmap.mmap | None = None
        self.size = 0
        self.expert_count = 0
        self.layer_to_base_slot: Dict[int, int] = {}
        self.layer_expert_to_slot: Dict[Tuple[int, int], int] = {}
        self.buffer_np: np.ndarray | None = None
        self.buffer_mx: Any = None
        self.is_locked = False
        try:
            self._open()
        except Exception:
            self.close()
            raise

    def _open(self) -> None:
        import mlx.core as mx

        self.fd = os.open(str(self.path), os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size

        if self.size < HEADER_SIZE:
            raise ValueError(
                f"Slab pack {self.path} is truncated: "
                f"got {self.size} bytes, expected at least {HEADER_SIZE}"
            )

        # Parse header
        hdr = os.pread(self.fd, HEADER_SIZE, 0)
        if len(hdr) != HEADER_SIZE:
            raise ValueError(
                f"Could not read the complete slab pack header in {self.path}"
            )
        magic, version, expert_count, stride, hdr_size, _, _ = struct.unpack_from(
            "<IIIII8sI", hdr, 0
        )
        if magic != HEADER_MAGIC or version != HEADER_VERSION:
            raise ValueError(f"Invalid slab pack header in {self.path}")
        if stride != RECORD_STRIDE:
            raise ValueError(f"Unsupported record stride {stride} in {self.path}")
        if hdr_size != HEADER_SIZE:
            raise ValueError(
                f"Unsupported header size {hdr_size} in {self.path}, "
                f"expected {HEADER_SIZE}"
            )
        if expert_count > MAX_DIRECTORY_ENTRIES:
            raise ValueError(
                f"Slab pack directory has {expert_count} entries in {self.path}, "
                f"but the {HEADER_SIZE}-byte header supports only "
                f"{MAX_DIRECTORY_ENTRIES}"
            )

        expected_size = HEADER_SIZE + expert_count * stride
        if self.size != expected_size:
            raise ValueError(
                f"Invalid slab pack size in {self.path}: got {self.size} bytes, "
                f"expected {expected_size} from header"
            )

        self._mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)

        self.expert_count = expert_count

        # Parse directory
        dir_offset = DIRECTORY_OFFSET
        seen_slots = set()
        seen_experts = set()
        for _ in range(expert_count):
            layer_id, expert_id, slot = struct.unpack_from("<HHI", hdr, dir_offset)
            if slot >= expert_count:
                raise ValueError(
                    f"Invalid slab pack directory slot {slot} in {self.path}: "
                    f"expected a value below {expert_count}"
                )
            if slot in seen_slots:
                raise ValueError(
                    f"Duplicate slab pack directory slot {slot} in {self.path}"
                )
            expert_key = (layer_id, expert_id)
            if expert_key in seen_experts:
                raise ValueError(
                    f"Duplicate slab pack directory expert {expert_key} in {self.path}"
                )
            seen_slots.add(slot)
            seen_experts.add(expert_key)
            if layer_id not in self.layer_to_base_slot:
                self.layer_to_base_slot[layer_id] = slot
            self.layer_expert_to_slot[(layer_id, expert_id)] = slot
            dir_offset += 8

        # Create zero-copy numpy array and wrap into MLX array via DLPack
        self.buffer_np = np.frombuffer(self._mm, dtype=np.uint8)
        self.buffer_mx = mx.from_dlpack(self.buffer_np)

        # Lock in physical RAM if requested
        if self.lock_memory:
            buf_ptr = self.buffer_np.__array_interface__["data"][0]
            if buf_ptr:
                self.is_locked = libc_mlock(buf_ptr, self.size)

    @property
    def allocation_digest(self) -> str:
        """Return a deterministic short digest of the validated directory."""
        digest = hashlib.sha256()
        digest.update(struct.pack("<II", self.expert_count, RECORD_STRIDE))
        for (layer_id, expert_id), slot in sorted(
            self.layer_expert_to_slot.items()
        ):
            digest.update(struct.pack("<HHI", layer_id, expert_id, slot))
        return digest.hexdigest()[:16]

    def close(self) -> None:
        if self._mm is not None:
            if self.is_locked and self.buffer_np is not None:
                buf_ptr = self.buffer_np.__array_interface__["data"][0]
                if buf_ptr:
                    libc_munlock(buf_ptr, self.size)
                self.is_locked = False
            self.buffer_mx = None
            self.buffer_np = None
            import gc
            gc.collect()
            try:
                self._mm.close()
            except BufferError:
                pass
            self._mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_slab_pack_cache_path(
    model_dir: str | Path,
    allocation: Mapping[int, Sequence[int]],
    cache_dir: str | Path | None = None,
) -> Path:
    """Compute deterministic cache path for a given model and slab allocation."""
    if cache_dir is None:
        cache_dir = Path(os.path.expanduser("~/.cache/flashnext"))
    else:
        cache_dir = Path(cache_dir).expanduser()

    # Hash model path + allocation structure
    alloc_str = ",".join(
        f"{l}:" + "-".join(map(str, sorted(allocation[l])))
        for l in sorted(allocation.keys())
    )
    key = f"{str(model_dir)}|{alloc_str}|v{HEADER_VERSION}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    total_slots = sum(len(v) for v in allocation.values())
    return cache_dir / f"slab-pack-slots{total_slots}-{digest}.bin"


def get_or_create_slab_pack(
    store: Any,
    allocation: Mapping[int, Sequence[int]],
    cache_dir: str | Path | None = None,
    lock_memory: bool = True,
) -> SlabPack:
    """Retrieve existing cached slab pack or build it once, then map and lock."""
    if not allocation:
        raise ValueError("Cannot create slab pack with empty allocation")

    cache_path = get_slab_pack_cache_path(store.dir, allocation, cache_dir)
    expected_path = os.environ.get("FLASHNEXT_SLAB_PACK_EXPECTED_PATH")
    if expected_path:
        expected = Path(expected_path).expanduser().resolve()
        if cache_path.resolve() != expected:
            raise RuntimeError(
                f"Resolved slab pack {cache_path} does not match prepared pack "
                f"{expected}"
            )
    if not cache_path.exists():
        if os.environ.get("FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING") == "1":
            raise FileNotFoundError(
                f"Required prebuilt slab pack is missing: {cache_path}. "
                "Run bench_slab_production.py --capacity-sweep --prepare-only, "
                "then reboot before the trusted benchmark."
            )
        build_slab_pack(store, allocation, cache_path)

    return SlabPack(cache_path, lock_memory=lock_memory)
