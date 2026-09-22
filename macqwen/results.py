"""The one place every test and benchmark writes its results.

Every run gets its own folder:

    results/<model>/<YYYYMMDD-HHMMSS>-<test-id>/
        record.jsonl   canonical measurement record (macqwen.measurement)
        output.log     the run's complete stdout and stderr
        ...            every other artifact the run writes

The test terminal creates the folder and hands it to the child process in
``MACQWEN_RESULTS_DIR``. A benchmark started by hand calls ``run_directory``
and gets its own folder under the same root. ``output_path`` refuses any
destination outside ``results/``, so a result cannot land in a cache, in
``/tmp`` or beside the code. Caches that runs read back as state, such as
pin profiles and slab packs, are not results and keep their own locations.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
ENVIRONMENT_KEY = "MACQWEN_RESULTS_DIR"
RECORD_NAME = "record.jsonl"
LOG_NAME = "output.log"


def slug(text: str) -> str:
    """Lowercase, dash-separated form of a test or script name."""
    words = str(text).strip().lower().replace("_", "-").split()
    return "-".join(word for word in words if word) or "run"


def model_root(model: str, root: Path | None = None) -> Path:
    base = RESULTS_ROOT if root is None else Path(root)
    return base / slug(model).replace("-", "_")


def new_run_directory(model: str, name: str, root: Path | None = None) -> Path:
    """Create a fresh, uniquely named run folder."""
    base = model_root(model, root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base / f"{stamp}-{slug(name)}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{slug(name)}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def run_directory(model: str, name: str) -> Path:
    """The folder this process writes to.

    Under the terminal this is the folder it created. A benchmark started by
    hand gets a new folder named after itself, created once per process.
    """
    handed = os.environ.get(ENVIRONMENT_KEY)
    if handed:
        path = ensure_inside(Path(handed))
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = new_run_directory(model, f"{name}-manual").resolve()
    os.environ[ENVIRONMENT_KEY] = str(path)
    return path


def ensure_inside(path: Path, root: Path | None = None) -> Path:
    """Return ``path`` resolved, or raise when it is outside the results root."""
    resolved = Path(path).expanduser().resolve()
    base = (RESULTS_ROOT if root is None else Path(root)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"results must be written under {base}; got {resolved}. "
            "Use macqwen.results.output_path()."
        ) from exc
    return resolved


def output_path(model: str, name: str, filename: str, explicit=None) -> Path:
    """Destination for one artifact.

    ``explicit`` is a path the caller was given, for example ``--json``. It is
    accepted only inside ``results/``. Without it the file goes in this run's
    folder.
    """
    if explicit:
        path = ensure_inside(Path(explicit))
    else:
        path = run_directory(model, name) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def latest(model: str, filename: str, root: Path | None = None) -> Path | None:
    """Newest ``filename`` from any earlier run of ``model``, or None.

    Lets one test consume the evidence another test recorded, such as a
    worker sweep that gates a later comparison.
    """
    base = model_root(model, root)
    if not base.is_dir():
        return None
    matches = [path for path in base.glob(f"*/{filename}") if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, str(path)))
