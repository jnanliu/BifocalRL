#!/usr/bin/env python3
"""Normalize a saved Qwen3.5 text checkpoint without touching tensor bytes.

The training pipeline may save the text backbone under the multimodal
container prefix ``model.language_model.*``.  vLLM's text-only architecture
expects ``model.*``.  Safetensors offsets are relative to the data section, so
the shorter keys fit inside the existing padded JSON header.  This tool backs
up that complete header, rewrites it in place, and leaves the tensor payload
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path


def _read_header(path: Path) -> tuple[int, bytes, dict, list[str]]:
    with path.open("rb") as f:
        raw_length = f.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors length prefix: {path}")
        header_len = struct.unpack("<Q", raw_length)[0]
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"Truncated safetensors header: {path}")
        header = json.loads(raw_header)
    return header_len, raw_header, header, [key for key in header if key != "__metadata__"]


def _normalize_key(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "model.", 1)
    if key.startswith("language_model."):
        return key.replace("language_model.", "model.", 1)
    if key.startswith(
        (
            "visual.",
            "vision.",
            "video.",
            "image.",
            "model.visual.",
            "model.vision.",
            "model.video.",
            "model.image.",
            "model.visual_merger.",
        )
    ):
        return None
    return key


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def repair_checkpoint(checkpoint_dir: Path, backup_root: Path) -> None:
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Expected config.json and model.safetensors in {checkpoint_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError(
            f"Refusing to repair non-text Qwen3.5 config in {checkpoint_dir}: "
            f"model_type={config.get('model_type')!r}"
        )

    original_size = weight_path.stat().st_size
    header_len, raw_header, header, old_keys = _read_header(weight_path)
    container_prefixes = ("model.language_model.", "language_model.")
    nested_keys = [key for key in old_keys if key.startswith(container_prefixes)]
    if not nested_keys:
        print(f"already_normalized={checkpoint_dir} keys={len(old_keys)}")
        return

    normalized_header = {}
    normalized_keys = []
    for key, value in header.items():
        if key == "__metadata__":
            normalized_header[key] = value
            continue
        new_key = _normalize_key(key)
        if new_key is None:
            continue
        if new_key in normalized_header:
            raise ValueError(f"Key collision while normalizing {weight_path}: {new_key}")
        normalized_header[new_key] = value
        normalized_keys.append(new_key)

    if any(key.startswith(container_prefixes) for key in normalized_keys):
        raise AssertionError("Normalization left Qwen3.5 container-prefixed keys")
    if not any(key.startswith("model.") for key in normalized_keys):
        raise AssertionError("Normalized checkpoint has no model.* weights")

    encoded_header = json.dumps(
        normalized_header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded_header) > header_len:
        raise ValueError(
            f"Normalized header grew beyond the safetensors header budget: "
            f"{len(encoded_header)} > {header_len}"
        )
    padded_header = encoded_header + (b" " * (header_len - len(encoded_header)))

    experiment_name = checkpoint_dir.parents[2].name
    backup_dir = backup_root / experiment_name / checkpoint_dir.name
    backup_path = backup_dir / "model.safetensors.header.backup"
    manifest_path = backup_dir / "manifest.json"
    original_prefix = struct.pack("<Q", header_len) + raw_header
    if backup_path.exists():
        if backup_path.read_bytes() != original_prefix:
            raise ValueError(f"Conflicting header backup already exists: {backup_path}")
    else:
        _atomic_write(backup_path, original_prefix)

    manifest = {
        "checkpoint": str(checkpoint_dir),
        "weight_file_size": original_size,
        "header_length": header_len,
        "tensor_count": len(old_keys),
        "renamed_key_count": len(nested_keys),
        "original_header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "normalized_header_sha256": hashlib.sha256(padded_header).hexdigest(),
    }
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    try:
        with weight_path.open("r+b", buffering=0) as f:
            f.seek(8)
            f.write(padded_header)
            os.fsync(f.fileno())

        new_header_len, _, written_header, new_keys = _read_header(weight_path)
        if new_header_len != header_len or weight_path.stat().st_size != original_size:
            raise AssertionError("Safetensors layout changed during header-only repair")
        if any(key.startswith(container_prefixes) for key in new_keys):
            raise AssertionError("Repaired checkpoint still has nested Qwen3.5 keys")
        if len(new_keys) != len(normalized_keys):
            raise AssertionError("Written checkpoint header does not match tensor count")
        for old_key, old_metadata in header.items():
            if old_key == "__metadata__":
                if written_header.get(old_key) != old_metadata:
                    raise AssertionError("Safetensors file metadata changed during repair")
                continue
            new_key = _normalize_key(old_key)
            if new_key is not None and written_header.get(new_key) != old_metadata:
                raise AssertionError(
                    f"Tensor metadata or data offsets changed during repair: {old_key}"
                )
    except Exception:
        with weight_path.open("r+b", buffering=0) as f:
            f.seek(0)
            f.write(original_prefix)
            os.fsync(f.fileno())
        raise

    print(
        f"repaired={checkpoint_dir} nested_keys={len(nested_keys)} "
        f"new_keys={len(new_keys)} header_backup={backup_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--backup-root",
        type=Path,
        required=True,
        help="Directory where original safetensors headers and manifests are stored.",
    )
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        repair_checkpoint(checkpoint, args.backup_root)


if __name__ == "__main__":
    main()
