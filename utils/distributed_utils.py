# coding=utf-8
"""Distributed / multi-process helpers shared by generate-and-score pipelines."""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from accelerate import Accelerator
except Exception:  # pragma: no cover
    Accelerator = None  # type: ignore[assignment]

from utils.generate import sleep_server_api, stop_vllm_server_on_port, is_port_in_use


def get_accelerator():
    """Best-effort create an ``Accelerator`` with extended NCCL timeout.

    We increase the process-group timeout because generation may run only on the
    main process while other ranks wait at barriers.  Controlled via
    ``CO_NCCL_TIMEOUT_SECONDS`` (default 3600 s = 1 h).
    """
    if Accelerator is None:
        return None
    try:
        timeout_s = int(os.environ.get("CO_NCCL_TIMEOUT_SECONDS", "3600"))
        if timeout_s <= 0:
            timeout_s = 3600
        from datetime import timedelta
        try:
            from accelerate.utils import InitProcessGroupKwargs  # type: ignore
            ipg = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_s))
            return Accelerator(kwargs_handlers=[ipg])
        except Exception:
            return Accelerator()
    except Exception:
        return None


def is_main_process(accelerator) -> bool:
    return accelerator is None or bool(getattr(accelerator, "is_main_process", True))


def wait_for_everyone(accelerator) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()


def sleep_vllm_server_if_possible(port: int, is_main: bool) -> None:
    """Best-effort: put an externally-started vLLM server to sleep to free VRAM.

    Important when generation is skipped due to cache hits, otherwise the server
    stays awake and can OOM the subsequent in-process scoring/logps steps.
    """
    if not is_main:
        return
    try:
        sleep_server_api("localhost", int(port), level=1)
    except Exception:
        pass


def stop_vllm_server_if_possible(port: int, is_main: bool) -> bool:
    """Best-effort: force-stop a vLLM server to fully free VRAM.

    Returns ``True`` if the port appears closed after our attempts.
    """
    if not is_main:
        return True
    port = int(port)
    try:
        ok = bool(stop_vllm_server_on_port("localhost", port, timeout_s=120.0))
        if (not ok) and bool(is_port_in_use(port)):
            ok = bool(stop_vllm_server_on_port("127.0.0.1", port, timeout_s=120.0))
        if (not ok) and bool(is_port_in_use(port)):
            try:
                sleep_server_api("localhost", port, level=2)
            except Exception:
                pass
        closed = not bool(is_port_in_use(port))
        if not closed:
            logger.warning(
                f"vLLM server still appears to be running on port {port} after stop attempts. "
                "This can OOM in-process scoring/logps. Please stop the server manually."
            )
        return bool(closed)
    except Exception:
        return not bool(is_port_in_use(port))


def shard_range(n: int, accelerator) -> Tuple[int, int, int]:
    """Return a contiguous shard ``(start, end, per_rank)`` for distributing *n* items.

    Uses fixed ``per_rank = ceil(n / world)`` so that gathered tensors have equal shapes.
    """
    if accelerator is None or int(getattr(accelerator, "num_processes", 1)) <= 1:
        return 0, n, n
    world = int(getattr(accelerator, "num_processes", 1))
    rank = int(getattr(accelerator, "process_index", 0))
    per_rank = (int(n) + world - 1) // world
    start = rank * per_rank
    end = min(start + per_rank, int(n))
    return start, end, per_rank
