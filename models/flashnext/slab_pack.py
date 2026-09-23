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
from contextlib import contextmanager
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

HEADER_MAGIC = 0x4D4F4553  # "MOES"
HEADER_VERSION = 1
HEADER_VERSION_G64 = 2
HEADER_SIZE = 4096
RECORD_STRIDE = 3072000
DIRECTORY_OFFSET = 32
DIRECTORY_ENTRY_SIZE = 8
MAX_DIRECTORY_ENTRIES = (HEADER_SIZE - DIRECTORY_OFFSET) // DIRECTORY_ENTRY_SIZE
_IDENTITY_VERSION = b"flashnext-slab-model-v1"

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


@dataclass(frozen=True)
class SlabLayout:
    """Immutable on-disk layout contract for one quantization group size."""

    name: str
    group_size: int
    record_stride: int
    header_version: int
    layout_id: int
    shapes: Tuple[Tuple[str, Tuple[int, int]], ...]
    offsets: Tuple[Tuple[str, int], ...]

    def shape(self, projection: str, part: str) -> Tuple[int, int]:
        return dict(self.shapes)[f"{projection}.{part}"]

    def offset(self, projection: str, part: str) -> int:
        return dict(self.offsets)[f"{projection}.{part}"]


def _make_layout(group_size: int) -> SlabLayout:
    if group_size == 32:
        metadata = 102400
        version, layout_id, name = HEADER_VERSION, 32, "q4-g32"
    elif group_size == 64:
        metadata = 51200
        version, layout_id, name = HEADER_VERSION_G64, 64, "q4-g64"
    else:
        raise ValueError(f"Unsupported slab quantization group size: {group_size}")
    gate_weight = 0
    gate_scales = 819200
    gate_biases = gate_scales + metadata
    up_weight = gate_biases + metadata
    up_scales = up_weight + 819200
    up_biases = up_scales + metadata
    down_weight = up_biases + metadata
    down_scales = down_weight + 819200
    down_biases = down_scales + metadata
    stride = down_biases + metadata
    if stride % 4096:
        raise AssertionError("Slab record stride must be page aligned")
    shapes = tuple(
        (f"{projection}.{part}", (rows, cols))
        for projection, rows, cols in (
            ("gate_proj", 640, 320),
            ("up_proj", 640, 320),
            ("down_proj", 2560, 80),
        )
        for part in _PARTS
    )
    # Weight columns are fixed. Metadata columns depend on group size.
    metadata_shapes = tuple(
        (f"{projection}.{part}", (rows, cols // group_size))
        for projection, rows, cols in (
            ("gate_proj", 640, 2560),
            ("up_proj", 640, 2560),
            ("down_proj", 2560, 640),
        )
        for part in ("scales", "biases")
    )
    shape_map = dict(shapes)
    shape_map.update(dict(metadata_shapes))
    offsets = (
        ("gate_proj.weight", gate_weight), ("gate_proj.scales", gate_scales),
        ("gate_proj.biases", gate_biases), ("up_proj.weight", up_weight),
        ("up_proj.scales", up_scales), ("up_proj.biases", up_biases),
        ("down_proj.weight", down_weight), ("down_proj.scales", down_scales),
        ("down_proj.biases", down_biases),
    )
    return SlabLayout(
        name=name, group_size=group_size, record_stride=stride,
        header_version=version, layout_id=layout_id,
        shapes=tuple(shape_map.items()), offsets=offsets,
    )


Q4G32_LAYOUT = _make_layout(32)
Q4G64_LAYOUT = _make_layout(64)


def get_slab_layout(layout: SlabLayout | str | int | None = None) -> SlabLayout:
    """Resolve a slab layout without changing the Q4/G32 default."""
    if layout is None:
        return Q4G32_LAYOUT
    if isinstance(layout, SlabLayout):
        if layout not in (Q4G32_LAYOUT, Q4G64_LAYOUT):
            raise ValueError(f"Unsupported slab layout descriptor: {layout.name}")
        return Q4G32_LAYOUT if layout == Q4G32_LAYOUT else Q4G64_LAYOUT
    if isinstance(layout, int):
        group_size = layout
    else:
        value = str(layout).lower().replace("_", "-")
        if value in {"g32", "q4g32", "q4-g32", "q4/g32"}:
            group_size = 32
        elif value in {"g64", "q4g64", "q4-g64", "q4/g64"}:
            group_size = 64
        else:
            raise ValueError(f"Unknown slab layout: {layout}")
    if group_size == 32:
        return Q4G32_LAYOUT
    if group_size == 64:
        return Q4G64_LAYOUT
    raise ValueError(f"Unsupported slab quantization group size: {group_size}")

_LIBC = None
_MLOCK = None
_MUNLOCK = None


@contextmanager
def _atomic_output(temp_path: Path, output_path: Path):
    """Write a cache file privately, then publish it atomically."""
    try:
        with open(temp_path, "wb") as handle:
            yield handle
        os.replace(temp_path, output_path)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


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
    layout: SlabLayout | str | int | None = None,
) -> int:
    """Extract expert weights from store and write a page-aligned slab pack.

    Returns the total file size in bytes.
    """
    slab_layout = get_slab_layout(layout)
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    if hasattr(store, "shape") and not validate_slab_allocation(
        store, allocation, layout=slab_layout
    ):
        raise ValueError("slab pack allocation contains an incompatible layer layout")
    total_size = HEADER_SIZE + expert_count * slab_layout.record_stride
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    with _atomic_output(temp_path, out_path) as f:
        # 1. Write Page 0 Header
        hdr = bytearray(HEADER_SIZE)
        struct.pack_into(
            "<IIIII8sI",
            hdr,
            0,
            HEADER_MAGIC,
            slab_layout.header_version,
            expert_count,
            slab_layout.record_stride,
            HEADER_SIZE,
            model_hash[:8].ljust(8, b"\x00"),
            0 if slab_layout is Q4G32_LAYOUT else slab_layout.layout_id,
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
            record_bytes = bytearray(slab_layout.record_stride)
            rec_offset = 0

            for proj in _PROJECTIONS:
                for part in _PARTS:
                    name = f"{prefix}.{proj}.{part}"
                    # Read single row as numpy array
                    row = store.rows_np(name, [expert_id])
                    expected_dtype = np.dtype(np.uint32 if part == "weight" else np.uint16)
                    if np.dtype(row.dtype) != expected_dtype:
                        raise ValueError(
                            f"Row dtype mismatch for {name}: got {row.dtype}, "
                            f"expected {expected_dtype}"
                        )
                    row_bytes = row.tobytes()
                    expected_bytes = (
                        slab_layout.shape(proj, part)[0]
                        * slab_layout.shape(proj, part)[1]
                        * row.dtype.itemsize
                    )
                    if len(row_bytes) != expected_bytes:
                        raise ValueError(
                            f"Row size mismatch for {name}: got {len(row_bytes)} bytes, "
                            f"expected {expected_bytes}"
                        )
                    part_offset = slab_layout.offset(proj, part)
                    record_bytes[part_offset : part_offset + len(row_bytes)] = row_bytes
                    rec_offset += len(row_bytes)

            if rec_offset != slab_layout.record_stride:
                raise ValueError(
                    f"Record size mismatch for layer {layer_id} expert {expert_id}: "
                    f"got {rec_offset} bytes, expected {slab_layout.record_stride}"
                )
            f.write(record_bytes)

    return total_size


def canonical_path(path: str | Path) -> Path:
    """Return ``path`` with symlinks resolved and its on-disk letter case.

    APFS is case-insensitive, so ``~/models`` and ``~/Models`` name the same
    directory while ``Path.resolve`` keeps whichever spelling it was given.
    Anything hashed from a path string must use the spelling the filesystem
    reports, or one checkpoint gets two identities.
    """
    resolved = Path(path).expanduser().resolve()
    command = getattr(fcntl, "F_GETPATH", None) if fcntl is not None else None
    if command is None:
        return resolved
    try:
        fd = os.open(str(resolved), os.O_RDONLY)
    except OSError:
        return resolved
    try:
        raw = fcntl.fcntl(fd, command, bytes(1024))
    except OSError:
        return resolved
    finally:
        os.close(fd)
    text = raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
    return Path(text) if text else resolved


def checkpoint_identity(model_dir: str | Path) -> str:
    """Return a cheap identity for the checkpoint used by a slab pack.

    The config and index are hashed by content. Referenced shard metadata uses
    file identity and size, so this never reads the checkpoint payload. The
    path is included in its on-disk spelling, so cache files never cross
    checkpoint locations but one location has one identity.
    """
    model_path = Path(model_dir).expanduser()
    digest = hashlib.sha256(_IDENTITY_VERSION)
    digest.update(str(canonical_path(model_path)).encode("utf-8"))
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("rb") as handle:
        index_bytes = handle.read()
    digest.update(index_path.name.encode("utf-8"))
    digest.update(index_bytes)
    index = json.loads(index_bytes)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"checkpoint index has no shards: {index_path}")
    config_path = model_path / "config.json"
    if config_path.is_file():
        digest.update(config_path.name.encode("utf-8"))
        digest.update(config_path.read_bytes())
    for shard_name in shards:
        shard_path = model_path / str(shard_name)
        stat = shard_path.stat()
        digest.update(str(shard_name).encode("utf-8"))
        digest.update(
            f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
            f"{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode("utf-8")
        )
    return digest.hexdigest()


def validate_slab_allocation(
    store: Any,
    allocation: Mapping[int, Sequence[int]],
    layout: SlabLayout | str | int | None = None,
) -> bool:
    """Check allocated layers against the selected packed-record contract."""
    slab_layout = get_slab_layout(layout)
    expected = {
        projection: {
            part: slab_layout.shape(projection, part)
            for part in _PARTS
        }
        for projection in _PROJECTIONS
    }
    try:
        expert_count = None
        for layer in allocation:
            experts = [int(expert) for expert in allocation[layer]]
            if len(set(experts)) != len(experts) or any(expert < 0 for expert in experts):
                return False
            prefix = f"language_model.model.layers.{int(layer)}.mlp.switch_mlp"
            for projection, parts in expected.items():
                for part, tail in parts.items():
                    name = f"{prefix}.{projection}.{part}"
                    shape = tuple(int(value) for value in store.shape(name))
                    if len(shape) != 3 or shape[1:] != tail:
                        return False
                    if expert_count is None:
                        expert_count = shape[0]
                    elif shape[0] != expert_count:
                        return False
                    ref = getattr(store, "refs", {}).get(name)
                    dtype = "BF16" if part != "weight" else "U32"
                    if ref is not None and ref.dtype != dtype:
                        return False
                    dtype_fn = getattr(store, "dtype", None)
                    if callable(dtype_fn):
                        actual_dtype = str(dtype_fn(name)).upper()
                        if dtype not in actual_dtype and not (
                            dtype == "U32" and "UINT32" in actual_dtype
                        ):
                            return False
        if expert_count is not None and any(
            int(expert) >= expert_count
            for experts in allocation.values()
            for expert in experts
        ):
            return False
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OSError):
        return False
    return True


class SlabPack:
    """Memory-mapped, mlocked slab pack providing a single zero-copy MTLBuffer."""

    __slots__ = (
        "path",
        "lock_memory",
        "fd",
        "_mm",
        "size",
        "layout",
        "expected_layout",
        "model_hash",
        "expected_model_hash",
        "expert_count",
        "layer_to_base_slot",
        "layer_expert_to_slot",
        "buffer_np",
        "buffer_mx",
        "is_locked",
    )

    def __init__(
        self,
        path: str | Path,
        lock_memory: bool = True,
        expected_model_hash: bytes | None = None,
        expected_layout: SlabLayout | str | int | None = None,
    ):
        self.path = Path(path).expanduser()
        self.lock_memory = lock_memory
        self.fd: int | None = None
        self._mm: mmap.mmap | None = None
        self.size = 0
        self.expected_layout = expected_layout
        self.layout = get_slab_layout(expected_layout)
        self.model_hash = b"\x00" * 8
        self.expected_model_hash = expected_model_hash
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
        magic, version, expert_count, stride, hdr_size, model_hash, reserved = struct.unpack_from(
            "<IIIII8sI", hdr, 0
        )
        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid slab pack header in {self.path}")
        if self.expected_layout is None and version == HEADER_VERSION_G64:
            self.layout = Q4G64_LAYOUT
        if version != self.layout.header_version:
            raise ValueError(
                f"Slab pack layout version {version} does not match "
                f"expected {self.layout.header_version} in {self.path}"
            )
        if version == HEADER_VERSION and reserved != 0:
            raise ValueError(f"Invalid Q4/G32 layout identity in {self.path}")
        if version == HEADER_VERSION_G64 and reserved != Q4G64_LAYOUT.layout_id:
            raise ValueError(f"Invalid Q4/G64 layout identity in {self.path}")
        if stride != self.layout.record_stride:
            raise ValueError(f"Unsupported record stride {stride} in {self.path}")
        if hdr_size != HEADER_SIZE:
            raise ValueError(
                f"Unsupported header size {hdr_size} in {self.path}, "
                f"expected {HEADER_SIZE}"
            )
        self.model_hash = model_hash
        if self.expected_model_hash is not None:
            expected = bytes(self.expected_model_hash)[:8].ljust(8, b"\x00")
            if model_hash != expected:
                raise ValueError(
                    f"Slab pack model identity does not match checkpoint in {self.path}"
                )
        if expert_count > MAX_DIRECTORY_ENTRIES:
            raise ValueError(
                f"Slab pack directory has {expert_count} entries in {self.path}, "
                f"but the {HEADER_SIZE}-byte header supports only "
                f"{MAX_DIRECTORY_ENTRIES}"
            )

        expected_size = HEADER_SIZE + expert_count * self.layout.record_stride
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
        digest.update(struct.pack("<II", self.expert_count, self.layout.record_stride))
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
    model_identity: str | None = None,
    layout: SlabLayout | str | int | None = None,
) -> Path:
    """Compute deterministic cache path for a given model and slab allocation."""
    if cache_dir is None:
        cache_dir = Path(os.path.expanduser("~/.cache/flashnext"))
    else:
        cache_dir = Path(cache_dir).expanduser()

    slab_layout = get_slab_layout(layout)
    if model_identity is None:
        model_identity = checkpoint_identity(model_dir)
    # Hash model identity + allocation structure. The path remains part of the
    # key to avoid sharing cache files across separate checkpoint locations.
    alloc_str = ",".join(
        f"{l}:" + "-".join(map(str, allocation[l]))
        for l in sorted(allocation.keys())
    )
    # Keep the historical G32 key byte-for-byte stable. G64 gets an explicit
    # layout suffix so caches can never cross the two record formats.
    layout_key = "" if slab_layout is Q4G32_LAYOUT else f"|layout={slab_layout.name}"
    key = f"{str(canonical_path(model_dir))}|{model_identity}|{alloc_str}|v{slab_layout.header_version}{layout_key}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(key).hexdigest()[:16]
    total_slots = sum(len(v) for v in allocation.values())
    return cache_dir / f"slab-pack-slots{total_slots}-{digest}.bin"


def get_or_create_slab_pack(
    store: Any,
    allocation: Mapping[int, Sequence[int]],
    cache_dir: str | Path | None = None,
    lock_memory: bool = True,
    layout: SlabLayout | str | int | None = None,
) -> SlabPack:
    """Retrieve existing cached slab pack or build it once, then map and lock."""
    if not allocation:
        raise ValueError("Cannot create slab pack with empty allocation")

    slab_layout = get_slab_layout(layout)
    model_identity = checkpoint_identity(store.dir)
    model_hash = bytes.fromhex(model_identity)[:8]
    cache_path = get_slab_pack_cache_path(
        store.dir, allocation, cache_dir, model_identity=model_identity,
        layout=slab_layout,
    )
    expected_path = os.environ.get("FLASHNEXT_SLAB_PACK_EXPECTED_PATH")
    if expected_path:
        expected = Path(expected_path).expanduser().resolve()
        if cache_path.resolve() != expected:
            raise RuntimeError(
                f"Resolved slab pack {cache_path} does not match prepared pack "
                f"{expected}"
            )
    require_existing = os.environ.get("FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING") == "1"
    if cache_path.exists():
        try:
            pack = SlabPack(
                cache_path,
                lock_memory=lock_memory,
                expected_model_hash=model_hash,
                expected_layout=slab_layout,
            )
        except (OSError, ValueError) as error:
            if require_existing:
                raise RuntimeError(
                    f"Existing slab pack is stale or invalid: {cache_path}: {error}"
                ) from error
        else:
            _mark_used_and_prune(cache_path)
            return pack
    if require_existing:
        raise FileNotFoundError(
            f"Required prebuilt slab pack is missing: {cache_path}. "
            "Run bench_slab_production.py --capacity-sweep --prepare-only, "
            "then use the benchmark file-cache purge and quiescence gate."
        )
    build_slab_pack(
        store, allocation, cache_path, model_hash=model_hash, layout=slab_layout
    )

    pack = SlabPack(
        cache_path,
        lock_memory=lock_memory,
        expected_model_hash=model_hash,
        expected_layout=slab_layout,
    )
    _mark_used_and_prune(cache_path)
    return pack


def _max_age_days() -> float:
    try:
        return float(os.environ.get("FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS", "14"))
    except ValueError:
        return 14.0


def mark_used_and_prune(current: Path, pattern: str) -> int:
    """Record that ``current`` was used and delete long-unused siblings.

    Opening a cache file sets its mtime, so mtime means last use. Files in the
    same directory matching ``pattern`` and unused for
    ``FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS`` (default 14, 0 disables) are removed.
    Age rather than count keeps every file a benchmark has just prepared.
    Returns the number of files removed.
    """
    try:
        os.utime(current)
    except OSError:
        pass
    days = _max_age_days()
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        candidates = list(current.parent.glob(pattern))
    except OSError:
        return 0
    for path in candidates:
        try:
            if path.samefile(current) or path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _mark_used_and_prune(current: Path) -> int:
    """Slab packs: every new allocation writes a new pack of about 180 MB."""
    return mark_used_and_prune(current, "slab-pack-slots*.bin")
