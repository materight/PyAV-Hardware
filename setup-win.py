import os
import sys
import platform
import subprocess
from pathlib import Path

import av
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


# -----------------------------------------------------------------------------
# Windows-specific configuration
# -----------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"

# On Windows, CUDA is often found via CUDA_PATH instead of CUDA_HOME
CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", None)
CUDA_ARCH_LIST = os.environ.get("CUDA_ARCH_LIST", "75,86")

# Manually specify where your FFmpeg (headers & libs) is installed on Windows.
# Adjust these paths to your local FFmpeg installation. For instance, if you
# installed FFmpeg with vcpkg, set these accordingly.
FFMPEG_INCLUDE_DIR = r"C:\Users\Administrator\Downloads\Programs\ffmpeg-n6.1-latest-win64-gpl-shared-6.1\include"
FFMPEG_LIB_DIR = r"C:\Users\Administrator\Downloads\Programs\ffmpeg-n6.1-latest-win64-gpl-shared-6.1\lib"

# Which FFmpeg libraries you want to link against (the .lib import libs)
FFMPEG_LIBRARIES = [
    "avcodec",
    "avutil",
    # Add "avformat", "swscale", etc. if you need them
]

# -----------------------------------------------------------------------------
# get_include_dirs / library logic
# -----------------------------------------------------------------------------

def get_extension_config_windows():
    """
    On Windows, we won't use pkg-config. Instead, manually specify
    include_dirs, library_dirs, and libraries for CUDA and FFmpeg.
    """
    include_dirs = []
    library_dirs = []
    libraries = []

    # Add FFmpeg includes/libs
    include_dirs.append(FFMPEG_INCLUDE_DIR)
    library_dirs.append(FFMPEG_LIB_DIR)
    libraries.extend(FFMPEG_LIBRARIES)

    # Add CUDA includes/libs if CUDA is found
    if CUDA_HOME:
        include_dirs.append(str(Path(CUDA_HOME) / "include"))
        # Typically on Windows x64: <CUDA_HOME>/lib/x64
        library_dirs.append(str(Path(CUDA_HOME) / "lib" / "x64"))
        # Example: link the needed libraries. Adjust as needed (e.g. nppicc, cudart, etc.).
        libraries.append("nppicc")
        libraries.append("cudart")
        libraries.append("nppif")
        libraries.append("nppc")
        libraries.append("nppig")
        libraries.append("nppim")
        libraries.append("nppist")
        libraries.append("nppisu")

    return include_dirs, library_dirs, libraries


class CustomBuildExt(_build_ext):
    """
    Custom build_ext class to handle .cu files via NVCC on Windows.
    """
    def build_extensions(self):
        if not CUDA_HOME:
            raise ValueError("Couldn't find CUDA. Please set `CUDA_HOME` or `CUDA_PATH` on Windows.")
        nvcc_path = str(Path(CUDA_HOME) / "bin" / "nvcc.exe")

        # Add .cu source extension
        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # If we're building .cu files, switch to NVCC
            if Path(src).suffix == ".cu":
                self.compiler.set_executable("compiler_so", nvcc_path)
                self.compiler.set_executable("compiler_cxx", nvcc_path)
                self.compiler.set_executable("compiler", nvcc_path)
                postargs = extra_postargs["nvcc"]
            else:
                # Otherwise, MSVC
                postargs = extra_postargs["msvc"]
            default_compile(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler._compile = _compile
        super().build_extensions()


# -----------------------------------------------------------------------------
# Gather sources & compile args
# -----------------------------------------------------------------------------

cuda_filepaths = [str(path) for path in Path("avcuda/cuda").glob("**/*.cu")]
cuda_arch_list = CUDA_ARCH_LIST.split(",")
cuda_arch_flags = []
for arch in cuda_arch_list:
    cuda_arch_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

# For safety, add a fallback arch. You can remove or adjust.
# (Compute_86 is Ampere, for example.)
cuda_arch_flags.extend(["-gencode", "arch=compute_86,code=compute_86"])

include_dirs, library_dirs, libraries = get_extension_config_windows()

# Example compiler args for NVCC on Windows (MSVC)
extra_compile_args = {
    "msvc": ["/std:c++17"],  # MSVC flags
    "nvcc": [
        "-c",
        "--compiler-options=/std:c++17",    # compile as C++17
        "--ptxas-options=-v",
        *cuda_arch_flags
    ]
}

# -----------------------------------------------------------------------------
# Build extension modules
# -----------------------------------------------------------------------------

extensions = []
for pyx in Path("avcuda").glob("**/*.pyx"):
    module_name = (
        str(pyx.with_suffix(""))
        .replace("/", ".")
        .replace("\\", ".")  # Windows path fix
    )
    extensions.extend(cythonize(
        Extension(
            module_name,
            sources=[str(pyx)] + cuda_filepaths,
            include_dirs=["avcuda", av.get_include()] + include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            # No runtime_library_dirs on Windows
            extra_compile_args=extra_compile_args,
        ),
        language_level=3,
        build_dir="build",
        include_path=[av.get_include()],
    ))

setup(
    name="avcuda",
    packages=find_packages(exclude=["build*"]),
    ext_modules=extensions,
    cmdclass={"build_ext": CustomBuildExt},
)
