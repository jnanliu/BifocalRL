"""Process timeouts shared by math grading and answer simplification."""

import atexit
import importlib
import logging
import multiprocessing
from functools import wraps

from pebble import ProcessPool
from pebble.concurrent import process

# Spawn avoids inheriting CUDA state; one lazy worker bounds CPU use per actor.
# Pebble terminates and replaces workers whose tasks exceed their deadline.
_POOL = ProcessPool(
    max_workers=1,
    context=multiprocessing.get_context("spawn"),
    initializer=importlib.import_module,
    initargs=("openrlhf.algorithms.bifocalrl.legacy.math",),
)
atexit.register(_POOL.join)
atexit.register(_POOL.stop)


def _timeout_decorator(timeout_seconds: int = 8, *, fallback=False):
    def decorator(func):
        run = process(pool=_POOL, timeout=timeout_seconds)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return run(*args, **kwargs).result()
            except TimeoutError:
                logging.getLogger(__name__).warning("%s exceeded its %s second timeout", func.__name__, timeout_seconds)
                return fallback(*args, **kwargs) if callable(fallback) else fallback

        return wrapper

    return decorator
