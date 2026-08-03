"""Make pip-installed CUDA libraries loadable.

CTranslate2 — the engine under faster-whisper — links against cuBLAS and cuDNN
at load time but does not ship them. They can be installed from PyPI
(``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12``), which puts the DLLs inside
``site-packages/nvidia/*/bin`` where the Windows loader will not look.

The symptom when this is missing is not a clean ImportError: the process stalls
or dies while constructing the model, with no diagnostic. Registering the
directories up front turns an unexplained hang into a working GPU or an honest
message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Subdirectories of the ``nvidia`` namespace package that contain runtime DLLs.
_LIBRARY_DIRS = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")


def _nvidia_root() -> Path | None:
    for entry in sys.path:
        candidate = Path(entry) / "nvidia"
        if candidate.is_dir():
            return candidate
    return None


def register_cuda_libraries() -> list[Path]:
    """Add pip-installed CUDA library directories to the loader search path.

    Returns the directories registered, which is empty on a CPU-only install —
    that is the normal case, not a failure.

    On Linux the dynamic loader resolves these through ``RPATH`` or
    ``LD_LIBRARY_PATH`` and this is a no-op; the DLL-directory API is
    Windows-only.
    """
    root = _nvidia_root()
    if root is None:
        return []

    registered: list[Path] = []
    for name in _LIBRARY_DIRS:
        directory = root / name / "bin"
        if not directory.is_dir():
            continue

        if hasattr(os, "add_dll_directory"):
            # Covers DLLs resolved when a Python extension module is imported.
            os.add_dll_directory(str(directory))
            # Not sufficient on its own: CTranslate2 loads cuBLAS and cuDNN
            # lazily on the first inference, via a plain LoadLibrary call that
            # consults PATH and ignores the added DLL directories. Omitting this
            # produces "Library cublas64_12.dll is not found" at the first
            # transcription, long after construction appeared to succeed.
            _prepend(os.environ, "PATH", directory)
        else:
            _prepend(os.environ, "LD_LIBRARY_PATH", directory)
        registered.append(directory)
    return registered


def _prepend(environment: os._Environ[str], variable: str, directory: Path) -> None:
    """Put ``directory`` at the front of a path-list environment variable."""
    existing = environment.get(variable, "")
    parts = [str(directory), *(p for p in existing.split(os.pathsep) if p)]
    # dict.fromkeys deduplicates while preserving order, so repeated calls are
    # idempotent rather than growing PATH without bound.
    environment[variable] = os.pathsep.join(dict.fromkeys(parts))
