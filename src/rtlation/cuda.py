"""CUDA-Bibliotheken unter Windows auffindbar machen."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_cuda_dll_dirs(root: Path) -> None:
    """Nimmt .tools/cuda/bin (cuBLAS, cuDNN aus setup.ps1) in den DLL-Suchpfad auf.

    Muss vor dem Laden von faster-whisper laufen. Unter Linux kommen die
    Bibliotheken aus dem System bzw. dem Docker-Image.
    """
    if sys.platform != "win32":
        return
    bin_dir = root / ".tools" / "cuda" / "bin"
    if bin_dir.is_dir():
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
