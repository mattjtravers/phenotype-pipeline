"""Logging configuration for the phenotype pipeline.

Routes log records by level so operators can read the two streams separately
when running the CLI locally:

- ``INFO`` and ``DEBUG`` → stdout
- ``WARNING``, ``ERROR``, ``CRITICAL`` → stderr

AWS Lambda captures both streams as CloudWatch Logs regardless of routing,
so this configuration is intended for the launcher CLI and any other locally
run entry points. The Lambda handler relies on Lambda's runtime handler and
should NOT call :func:`configure_logging` at module load.

See ``docs/high-level-design.md § Cross-Cutting Code Standards`` for the policy
this module implements.
"""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Handler names — used to identify previously installed handlers across
# repeated configure_logging() calls in long-running test sessions. Without
# these, calling configure_logging() twice would attach duplicate handlers
# and emit each log line twice.
_STDOUT_HANDLER_NAME = "phenotype-pipeline-stdout"
_STDERR_HANDLER_NAME = "phenotype-pipeline-stderr"


def _stdout_only_filter(record: logging.LogRecord) -> bool:
    """Return True if the record should be written to stdout.

    Args:
        record: The log record under consideration.

    Returns:
        True if ``record.levelno`` is below ``logging.WARNING``; False otherwise.
    """
    return record.levelno < logging.WARNING


def configure_logging(level: int = logging.INFO) -> None:
    """Install stdout/stderr stream handlers on the root logger.

    The function is idempotent: any handlers previously installed by this
    module are removed before fresh handlers are attached. Handlers installed
    by other parties (pytest's ``caplog``, AWS Lambda's runtime, third-party
    libraries) are left in place.

    Args:
        level: Minimum severity to emit. Defaults to ``logging.INFO``.
            Records below this level are dropped before the per-handler
            filter runs.

    Returns:
        None. Side effect: mutates the root logger's handler list.
    """
    root = logging.getLogger()

    # Remove our previously installed handlers so a second configure_logging()
    # call doesn't double-emit. Other handlers (pytest caplog, Lambda runtime)
    # are deliberately left untouched.
    for handler in list(root.handlers):
        if handler.name in (_STDOUT_HANDLER_NAME, _STDERR_HANDLER_NAME):
            root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.name = _STDOUT_HANDLER_NAME
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(_stdout_only_filter)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.name = _STDERR_HANDLER_NAME
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)

    # Only lower the root level; never raise it, since other code may have
    # already set a more verbose level (e.g., DEBUG for troubleshooting).
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
