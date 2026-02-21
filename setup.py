"""Build script that drives CMake to compile pybind11 extensions.

This setup.py intentionally keeps metadata in pyproject.toml and only
implements a CMake-backed build_ext so `pip install .` works consistently.
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension that is built by CMake.

    Parameters
    - name: Python import path for the built extension (e.g. "cripser._cripser").
    - target: CMake target name to build (e.g. "_cripser").
    - sourcedir: CMake source directory (default: project root).
    """

    def __init__(self, name: str, *, target: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.target = target
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Invoke CMake to configure and build the requested target, then copy it
    to the correct `build_lib` location for packaging.
    """

    def initialize_options(self) -> None:  # type: ignore[override]
        super().initialize_options()
        self._configured = False
        self._build_temp = None

    def run(self) -> None:  # type: ignore[override]
        # Ensure CMake is available
        try:
            out = subprocess.check_output(["cmake", "--version"])  # noqa: S603,S607
        except OSError as exc:  # pragma: no cover
            raise RuntimeError("CMake is required to build the extensions") from exc
        # Proceed with normal build_ext flow
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:  # type: ignore[override]
        assert isinstance(ext, CMakeExtension)

        # Compute build and output dirs
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure CMake only once per build directory
        if not getattr(self, "_configured", False):
            cfg = "Debug" if self.debug else "Release"
            # Determine package version from pyproject.toml for injection
            version = os.environ.get("CRIPSER_VERSION")
            if not version:
                try:
                    if sys.version_info >= (3, 11):
                        import tomllib  # type: ignore[attr-defined]
                    else:  # pragma: no cover
                        import tomli as tomllib  # type: ignore
                    with open("pyproject.toml", "rb") as f:
                        data = tomllib.load(f)
                    version = data.get("project", {}).get("version", "dev")
                except Exception:
                    version = "dev"
            configure_cmd: List[str] = [
                "cmake",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_temp}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_temp}",
                f"-DCMAKE_BUILD_TYPE={cfg}",
                "-DPython_EXECUTABLE=" + sys.executable,
                "-DPython3_EXECUTABLE=" + sys.executable,
                f"-DCRIPSER_VERSION={version}",
            ]
            if sys.platform == "darwin":
                deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
                if deployment_target:
                    configure_cmd.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}")

                # cbuildwheel sets ARCHFLAGS (e.g. "-arch x86_64" / "-arch arm64 -arch x86_64");
                # propagate this to CMake so wheel tags and binary slices stay consistent.
                arch_tokens = shlex.split(os.environ.get("ARCHFLAGS", ""))
                archs = [
                    arch_tokens[i + 1]
                    for i, token in enumerate(arch_tokens[:-1])
                    if token == "-arch"
                ]
                if archs:
                    configure_cmd.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")

            configure_cmd.append(ext.sourcedir)
            subprocess.check_call(configure_cmd, cwd=build_temp)  # noqa: S603,S607
            self._configured = True

        # Build the specific CMake target
        build_cmd = [
            "cmake",
            "--build",
            ".",
            "--target",
            ext.target,
            "--config",
            "Debug" if self.debug else "Release",
        ]
        subprocess.check_call(build_cmd, cwd=build_temp)  # noqa: S603,S607

        # Locate the built artifact in the build_temp directory (search recursively for MSVC Release/)
        built_candidates: List[str] = []
        for pat in (f"{ext.target}*.so", f"{ext.target}*.pyd", f"{ext.target}*.dylib"):
            built_candidates.extend([str(p) for p in build_temp.rglob(pat)])
        if not built_candidates:
            raise RuntimeError(f"Could not find built artifact for {ext.target} in {build_temp}")

        built_path = Path(sorted(built_candidates, key=len)[-1])  # pick the most specific suffix

        # Ensure destination directory exists
        extdir.mkdir(parents=True, exist_ok=True)

        # Copy to the exact expected python extension path (including ABI suffix)
        shutil.copy2(built_path, ext_fullpath)
        self.announce(f"Copied {built_path.name} -> {ext_fullpath}", level=3)


if __name__ == "__main__":
    setup(
        ext_modules=[
            # Place _cripser inside the "cripser" package
            CMakeExtension("cripser._cripser", target="_cripser"),
            # Expose tcripser as a top-level module
            CMakeExtension("tcripser", target="tcripser"),
        ],
        cmdclass={"build_ext": CMakeBuild},
        packages=[
            "cripser*",
        ],
        # Avoid copying arbitrary files (e.g., egg-info) into wheels on Windows
        include_package_data=False,
        zip_safe=False,
    )
