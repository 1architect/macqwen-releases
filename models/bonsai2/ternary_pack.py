"""Derived base-243 gate/up sidecar: repack, fingerprint, load.

The checkpoint stays immutable.  The repacker (offline script, not
here) converts gate/up Q2 codes to 5 trits per byte with 26 bytes per
128-weight group.  This module loads that sidecar, validates its
header against the checkpoint fingerprint, and hands ternary bytes
to the gate/up decoder.  Missing or mismatched sidecar fails closed
to the stock 2-bit path.
"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

MAGIC = b"B243GATEUP"
VERSION = 2
GROUPS_PER_ROW = 40
PADDED_BYTES_PER_GROUP = 26
WIDTH = 5120
ROWS = 17408


def checkpoint_fingerprint(checkpoint: str | os.PathLike[str]) -> str:
    root = Path(checkpoint).expanduser()
    digest = hashlib.sha256()
    digest.update((root / "config.json").read_bytes())
    st = (root / "model.safetensors").stat()
    digest.update(struct.pack("<QQQ", st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    digest.update(b"ternary243-v1")
    return digest.hexdigest()


def sidecar_path(checkpoint: str | os.PathLike[str]) -> Path:
    digest = checkpoint_fingerprint(checkpoint)
    return (
        Path.home()
        / ".cache"
        / "bonsai2"
        / f"ternary-gateup-{digest[:16]}-v{VERSION}.bin"
    )


def load_gateup(checkpoint: str | os.PathLike[str]) -> dict[str, object]:
    """Load and validate the gate/up ternary sidecar.

    Returns module paths plus raw tensors.  Raises on any mismatch so
    callers fall back to the stock path.
    """
    import mlx.core as mx

    root = Path(checkpoint).expanduser()
    config = __import__("json").loads((root / "config.json").read_text())
    names = [
        m["path"]
        for m in config["modules"]
        if m["path"].split(".")[-1] in ("gate_proj", "up_proj")
        and "mlp" in m["path"]
        and not m.get("embedding")
    ]
    if len(names) != 128:
        raise ValueError(f"expected 128 gate/up modules, found {len(names)}")
    path = sidecar_path(checkpoint)
    with open(path, "rb") as handle:
        header = handle.read(18)
        if header != MAGIC + struct.pack("<II", VERSION, 128):
            raise ValueError("ternary sidecar magic/version/count mismatch")
        if handle.read(32) != bytes.fromhex(checkpoint_fingerprint(checkpoint)):
            raise ValueError("ternary sidecar checkpoint mismatch")
        per_row = GROUPS_PER_ROW * PADDED_BYTES_PER_GROUP
        modules = {}
        for name in names:
            rows, width = struct.unpack("<II", handle.read(8))
            if (rows, width) != (ROWS, WIDTH):
                raise ValueError(f"ternary sidecar geometry mismatch at {name}")
            raw = handle.read(rows * per_row)
            if len(raw) != rows * per_row:
                raise ValueError(f"ternary sidecar truncated at {name}")
            modules[name] = mx.array(bytearray(raw), dtype=mx.uint8).reshape(
                rows, per_row
            )
    return {"path": str(path), "digest": checkpoint_fingerprint(checkpoint), "modules": modules}
