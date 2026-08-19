"""Run the per-document parsing stages across cores.

Every heavy stage in this pipeline has the same shape: a loop over independent
documents, each producing a small frame, concatenated at the end.  Parsing
1,666 XBRL instances, 1,004 investor documents and 210 filing HTML bodies is
around eleven minutes of one core, on machines that have twenty.

Two properties have to survive the move to a pool, and both are load-bearing
for a database whose output is meant to be reproducible:

**Order.**  ``Executor.map`` yields results in *input* order, not completion
order, so the concatenated frame is identical to the serial one rather than
merely equivalent.  A panel that reshuffles between builds would make every
diff unreadable.

**Failure.**  One unparseable document must not take the run down, exactly as
in the serial loops this replaces.  Workers return the exception rather than
raising it, and the parent logs it with the same message the serial path used,
so a bad document still shows up in the build output.

Fetching stays in the parent.  The rate limiter in :mod:`edgar` spaces requests
within a single process, and a pool of workers each holding their own limiter
would multiply the request rate by the worker count -- straight through the
SEC's fair-access limit.  Callers therefore warm the cache serially first and
the workers only ever read from disk.
"""

from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

# Capped well below the core count on purpose: these workers hold a parsed XML
# tree, and JPMorgan's instance document alone is 27 MB of source that lxml
# expands by roughly an order of magnitude.  Override with BANKQTR_WORKERS=1
# to get the serial path back, which is the first thing to try when a parallel
# build disagrees with a serial one.
DEFAULT_MAX_WORKERS = 8


def max_workers() -> int:
    env = os.environ.get("BANKQTR_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            log.warning("ignoring non-numeric BANKQTR_WORKERS=%r", env)
    return max(1, min(DEFAULT_MAX_WORKERS, (os.cpu_count() or 2) - 1))


# One pool for the whole process.  ``dimensional_long`` runs per bank, so a
# pool built per call is built 31 times in a universe run -- and Windows
# *spawns* workers rather than forking, so each one re-imports polars and lxml.
# That start-up cost was eating most of what the parallelism won.
_POOL: ProcessPoolExecutor | None = None


def _shared_pool(workers: int) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers)
        atexit.register(shutdown)
    return _POOL


def shutdown() -> None:
    """Release the shared pool.  Registered at exit; safe to call twice."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _run(args: tuple[Callable[..., pl.DataFrame], Any]) -> Any:
    """Worker side: never raise, so one bad document cannot end the run."""
    fn, item = args
    try:
        return fn(item)
    except Exception as exc:  # noqa: BLE001 - reported by the parent
        return exc


def map_frames[T](
    fn: Callable[[T], pl.DataFrame],
    items: Iterable[T],
    *,
    schema: Any,
    label: str = "parse",
    describe: Callable[[T], str] | None = None,
) -> pl.DataFrame:
    """Apply ``fn`` to each item across a pool, concatenated in input order.

    ``fn`` must be a module-level function taking one picklable argument:
    Windows spawns rather than forks, so a closure or a lambda cannot cross the
    boundary.
    """
    work = list(items)
    if not work:
        return pl.DataFrame(schema=schema)

    # Sized from the machine, never from this batch: the pool is shared, and a
    # first call with two items must not leave every later call with two
    # workers.  A batch too small to be worth the hand-off stays inline.
    workers = max_workers()
    if workers <= 1 or len(work) < 2:
        results: list[Any] = [_run((fn, item)) for item in work]
    else:
        # chunksize amortises the per-item IPC round trip; these items are
        # tens to hundreds of milliseconds each, so a small chunk is enough.
        pool = _shared_pool(workers)
        results = list(pool.map(_run, ((fn, item) for item in work), chunksize=4))

    frames: list[pl.DataFrame] = []
    for item, result in zip(work, results, strict=True):
        if isinstance(result, Exception):
            name = describe(item) if describe else repr(item)
            log.warning("%s failed %s: %s", label, name, result)
            continue
        if result is not None and not result.is_empty():
            frames.append(result)

    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames, how="vertical_relaxed")
