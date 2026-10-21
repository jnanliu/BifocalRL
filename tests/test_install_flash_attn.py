"""Verify wheel compatibility and installation without fetching GPU binaries."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.tags import Tag


@pytest.fixture
def installer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/scripts/install_flash_attn.py"
    spec = importlib.util.spec_from_file_location("install_flash_attn", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(cuda="13.0", hip=None)),
    )
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
@pytest.mark.parametrize("encoded", [False, True])
def test_selects_exact_version_cuda_torch_python_and_architecture(installer, monkeypatch, architecture, encoded):
    prefix = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.48/"
    expected = prefix + f"flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_{architecture}.whl"
    alternatives = [
        expected.replace("2.8.3+", "2.8.4+"),
        expected.replace("cu130", "cu128"),
        expected.replace("torch2.13", "torch2.12"),
        expected.replace("cp312", "cp311"),
        expected.replace(architecture, "aarch64" if architecture == "x86_64" else "x86_64"),
    ]
    index = "\n".join(f"[Download]({url.replace('+', '%2B') if encoded else url})" for url in [expected, *alternatives])
    monkeypatch.setattr(installer, "_read_url", lambda url: index)
    monkeypatch.setattr(installer, "sys_tags", lambda: [Tag("cp312", "cp312", f"linux_{architecture}")])
    assert installer.find_wheel_url() == expected


def test_missing_wheel_fails_without_installing_or_compiling(installer, monkeypatch):
    monkeypatch.setattr(installer, "_read_url", lambda url: "No matching wheels")
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected installation"))
    with pytest.raises(RuntimeError, match="No matching flash-attn wheel"):
        installer.install_flash_attn()


@pytest.mark.parametrize("cuda,hip", [(None, None), (None, "7.2")])
def test_cpu_and_rocm_are_rejected(installer, cuda, hip):
    installer.torch.version.cuda = cuda
    installer.torch.version.hip = hip
    with pytest.raises(RuntimeError):
        installer.find_wheel_url()


@pytest.mark.parametrize(
    "version,valid",
    [
        ("2.8.3", True),
        ("2.8.3+cu130torch2.13", True),
        ("2.8.4+cu130torch2.13", False),
        ("2.8.3+cu128torch2.13", False),
        ("2.8.3+cu130torch2.12", False),
    ],
)
def test_existing_install_must_match_version_and_runtime(installer, monkeypatch, version, valid):
    monkeypatch.setattr(installer, "package_version", lambda name: version)
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())
    assert installer.check_flash_attn_installed() is valid


@pytest.mark.parametrize("uv", [False, True])
@pytest.mark.parametrize("keep_wheel", [False, True])
def test_install_preserves_dependencies_and_wheel_retention(installer, monkeypatch, tmp_path, uv, keep_wheel):
    url = (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.48/"
        "flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_aarch64.whl"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, "find_wheel_url", lambda: url)
    monkeypatch.setattr(installer, "_download_file", lambda source, target: Path(target).write_bytes(b"wheel"))
    monkeypatch.setattr(installer, "check_flash_attn_installed", lambda: True)
    commands = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, check: commands.append(command))
    installer.install_flash_attn(uv=uv, keep_wheel=keep_wheel)
    assert len(commands) == 1
    expected = (
        ["uv", "pip", "install", "--python", installer.sys.executable, "--no-deps"]
        if uv
        else [installer.sys.executable, "-m", "pip", "install", "--no-deps"]
    )
    assert commands[0][:-1] == expected
    wheel = Path(commands[0][-1])
    assert wheel.exists() is keep_wheel
    if keep_wheel:
        assert wheel.parent == tmp_path


def test_valid_existing_install_skips_download(installer, monkeypatch):
    monkeypatch.setattr(installer, "check_flash_attn_installed", lambda: True)
    monkeypatch.setattr(installer, "_read_url", lambda url: pytest.fail("unexpected download"))
    installer.main(uv=False, keep_wheel=False)
