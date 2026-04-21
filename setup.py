from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise RuntimeError(
        "Cython is required to build the optional CTC alignment extension. "
        "Create a local build environment and install `Cython` first."
    ) from exc

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    BuildExtension = None
    CUDAExtension = None


ROOT = Path(__file__).parent


def _relpath(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


CYTHON_EXTENSIONS = [
    Extension(
        "paraformer_v2._ctc_alignment_cython",
        [_relpath(ROOT / "src" / "paraformer_v2" / "_ctc_alignment_cython.pyx")],
        include_dirs=[np.get_include()],
    )
]


def _nvcc_version() -> str | None:
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
    except Exception:
        return None
    for line in output.splitlines():
        if "release " in line:
            return line.split("release ", 1)[1].split(",", 1)[0].strip()
    return None


def _should_build_cuda_extension() -> bool:
    if os.environ.get("ENABLE_CUDA_CTC_ALIGNMENT", "1") == "0":
        print("Skipping CUDA CTC alignment extension because ENABLE_CUDA_CTC_ALIGNMENT=0")
        return False
    if CUDAExtension is None or BuildExtension is None:
        print("Skipping CUDA CTC alignment extension because PyTorch C++ extension helpers are unavailable")
        return False
    try:
        import torch
    except ImportError:
        print("Skipping CUDA CTC alignment extension because torch is not importable during build")
        return False
    if torch.version.cuda is None:
        print("Skipping CUDA CTC alignment extension because this PyTorch build has no CUDA support")
        return False
    nvcc_version = _nvcc_version()
    if nvcc_version is None:
        print("Skipping CUDA CTC alignment extension because nvcc was not found")
        return False
    if nvcc_version != torch.version.cuda:
        print(
            "Skipping CUDA CTC alignment extension because nvcc "
            f"{nvcc_version} does not match torch CUDA {torch.version.cuda}"
        )
        return False
    return True


ext_modules = cythonize(
    CYTHON_EXTENSIONS,
    compiler_directives={"language_level": "3"},
)
cmdclass = {}

if _should_build_cuda_extension():
    ext_modules.append(
        CUDAExtension(
            "paraformer_v2._ctc_alignment_cuda",
            [
                _relpath(ROOT / "src" / "paraformer_v2" / "_ctc_alignment_cuda.cpp"),
                _relpath(ROOT / "src" / "paraformer_v2" / "_ctc_alignment_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
