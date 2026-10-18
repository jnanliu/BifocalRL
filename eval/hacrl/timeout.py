"""Lightweight timeout decorator for scoring functions."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from functools import wraps


def timeout_limit(seconds: float):
    """Decorator that aborts a function call after *seconds*."""

    def decorator(func):
        if os.name == "posix":

            @wraps(func)
            def wrapper(*args, **kwargs):
                def _handler(signum, frame):
                    raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

                old = signal.signal(signal.SIGALRM, _handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)

        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                ctx = mp.get_context("spawn")
                q: mp.Queue = ctx.Queue()

                def _target():
                    try:
                        q.put((True, func(*args, **kwargs)))
                    except Exception as e:
                        q.put((False, e))

                p = ctx.Process(target=_target)
                p.start()
                p.join(timeout=seconds)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
                if q.empty():
                    raise RuntimeError(f"{func.__name__} terminated without result")
                ok, val = q.get_nowait()
                if ok:
                    return val
                raise val

        return wrapper

    return decorator
