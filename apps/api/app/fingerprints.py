from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

RECOGNIZER_PKG_DIR = Path(__file__).resolve().parent / "transcription"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@lru_cache(maxsize=1)
def recognizer_fingerprint() -> str:
    """SHA-256 of all .py files under apps/api/app/transcription/ (first 16 hex chars).

    Edit any recognizer source → fingerprint changes. Computed once per process.
    """
    h = hashlib.sha256()
    for path in sorted(RECOGNIZER_PKG_DIR.rglob("*.py")):
        h.update(path.relative_to(RECOGNIZER_PKG_DIR).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def kalimba_dsp_fingerprint() -> str:
    """SHA-256 of the loaded kalimba_dsp extension binary (first 16 hex chars).

    Returns "absent" if not importable (older checkouts without Rust extension).
    """
    try:
        import kalimba_dsp  # type: ignore[import-not-found]
    except ImportError:
        return "absent"
    so_path_str = getattr(kalimba_dsp, "__file__", None)
    if not so_path_str:
        return "absent"
    so_path = Path(so_path_str)
    if not so_path.is_file():
        return "absent"
    try:
        with so_path.open("rb") as fh:
            digest = hashlib.file_digest(fh, "sha256")
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def git_head_sha() -> str | None:
    """Current git HEAD SHA of the repo (full 40 char). None if unavailable.

    Cached once per process. Restart the service after a new commit for an
    updated value (matches how the rest of the service reads its state at
    startup, e.g. allowed origins).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        return None
    return sha
