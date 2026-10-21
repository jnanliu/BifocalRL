"""Install a compatible third-party flash-attn wheel from a community index."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen

import torch
import typer
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

# Match requirements.txt so later installs keep the selected wheel.
FLASH_ATTN_RANGE = SpecifierSet("==2.8.3")
WHEEL_INDEX_URL = (
    "https://raw.githubusercontent.com/mjun0812/"
    "flash-attention-prebuild-wheels/main/doc/packages.md"
)
WHEEL_URL_PATTERN = re.compile(
    r"(https://github\.com/mjun0812/flash-attention-prebuild-wheels/"
    r"releases/download/[^ )]+/flash_attn-[^ )]+\.whl)"
)
DOWNLOAD_ATTEMPTS = 5


def _open_url(url: str):
    request = Request(url, headers={"User-Agent": "bifocalrl-flash-attn-installer"})
    return urlopen(request, timeout=60)


def _read_url(url: str) -> str:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with _open_url(url) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"Failed to download {url}") from error
            time.sleep(attempt)
    raise AssertionError("unreachable")


def _download_file(url: str, destination: str) -> None:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with _open_url(url) as response, open(destination, "wb") as output:
                shutil.copyfileobj(response, output)
            return
        except (HTTPError, URLError, TimeoutError) as error:
            if os.path.exists(destination):
                os.unlink(destination)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"Failed to download {url}") from error
            time.sleep(attempt)


def _get_cuda_torch_tag():
    # torch.version.hip/cuda are runtime attributes not in type stubs
    is_rocm = (
        hasattr(torch.version, "hip")
        and torch.version.hip is not None  # type: ignore[attr-defined]
    )
    if is_rocm:
        raise RuntimeError("ROCm wheels are not supported.")

    torch_cuda_version = torch.version.cuda  # type: ignore[attr-defined]
    if torch_cuda_version is None:
        raise RuntimeError("The installed PyTorch does not include CUDA support.")

    torch_major, torch_minor = torch.__version__.split("+", 1)[0].split(".")[:2]
    return f"cu{torch_cuda_version.replace('.', '')}torch{torch_major}.{torch_minor}"


def check_flash_attn_installed():
    try:
        installed_version = package_version("flash-attn")
        print(f"flash_attn version: {installed_version}")
        version = Version(installed_version)
        if version not in FLASH_ATTN_RANGE:
            return False
        if re.fullmatch(r"cu\d+torch\d+\.\d+", version.local or "") and (
            version.local != _get_cuda_torch_tag()
        ):
            return False
        __import__("flash_attn")
        return True
    except (ImportError, InvalidVersion, OSError, PackageNotFoundError, RuntimeError):
        return False


def find_wheel_url():
    cuda_torch_tag = _get_cuda_torch_tag()

    print(f"Detected: Python {sys.version_info.major}.{sys.version_info.minor} {cuda_torch_tag}")

    wheel_index = _read_url(WHEEL_INDEX_URL)
    supported_tags = set(sys_tags())
    matches = []
    for wheel_url in WHEEL_URL_PATTERN.findall(wheel_index):
        wheel_url = unquote(wheel_url)
        try:
            distribution, version, _, wheel_tags = parse_wheel_filename(
                wheel_url.rsplit("/", 1)[-1]
            )
        except InvalidWheelFilename:
            continue
        if (
            distribution == "flash-attn"
            and version in FLASH_ATTN_RANGE
            and version.local == cuda_torch_tag
            and supported_tags.intersection(wheel_tags)
        ):
            matches.append((version, wheel_url))
    if matches:
        return max(matches, key=lambda item: item[0])[1]

    raise RuntimeError(f"No matching flash-attn wheel found for {cuda_torch_tag}.")


def install_flash_attn(uv: bool = False, keep_wheel: bool = False):
    wheel_url = find_wheel_url()
    local_filename = wheel_url.rsplit("/", 1)[-1]

    print(f"wheel_url: {wheel_url}")

    def _install_helper(local_path: str):
        _download_file(wheel_url, local_path)
        install_cmd = (
            ["uv", "pip", "install", "--python", sys.executable, "--no-deps", local_path]
            if uv
            else [sys.executable, "-m", "pip", "install", "--no-deps", local_path]
        )
        subprocess.run(install_cmd, check=True)

    if keep_wheel:
        local_path = os.path.abspath(local_filename)
        _install_helper(local_path)
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            local_path = os.path.join(tempdir, local_filename)
            _install_helper(local_path)

    # Try to import flash_attn
    if not check_flash_attn_installed():
        print("Failed to install flash_attn.")
        sys.exit(1)


def main(
    uv: bool = typer.Option(False, help="Use uv pip to install instead of pip"),
    keep_wheel: bool = typer.Option(
        False, help="Keep the downloaded wheel file in current directory"
    ),
):
    """Install flash-attn from a matching pre-built wheel."""
    if check_flash_attn_installed():
        print("flash_attn is already installed. Skipping installation.")
        return
    install_flash_attn(uv=uv, keep_wheel=keep_wheel)


if __name__ == "__main__":
    typer.run(main)
