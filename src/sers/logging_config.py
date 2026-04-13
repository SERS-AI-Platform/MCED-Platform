"""
Centralized logging configuration for the SERS pipeline.

Provides JSON structured logging with rotation and configurable log levels.

Usage:
    from sers.logging_config import setup_logging
    setup_logging()  # call once at application entry point

    # Individual modules just use standard logging:
    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


# Default configuration
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
DEFAULT_LOG_FILE = "sers_pipeline.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LEVEL = "INFO"


def setup_logging(
    level: str | None = None,
    log_dir: Path | str | None = None,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    json_format: bool = True,
    console: bool = True,
) -> None:
    """Configure logging for the SERS pipeline.

    Call this once at application startup. All modules using
    ``logging.getLogger(__name__)`` will inherit this configuration.

    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR). Defaults to the
        ``SERS_LOG_LEVEL`` environment variable, or ``"INFO"``.
    log_dir : Path or str, optional
        Directory for log files. Defaults to ``<project_root>/logs/``.
        Set to ``None`` to disable file logging.
    log_file : str
        Log file name within *log_dir*.
    max_bytes : int
        Maximum log file size before rotation (default 10 MB).
    backup_count : int
        Number of rotated backups to keep (default 5).
    json_format : bool
        If True, use JSON structured format for the file handler.
        Console always uses a human-readable format.
    console : bool
        If True, add a StreamHandler for console output.
    """
    resolved_level = (
        level
        or os.environ.get("SERS_LOG_LEVEL")
        or DEFAULT_LEVEL
    ).upper()

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls
    if getattr(root_logger, "_sers_logging_configured", False):
        return
    root_logger._sers_logging_configured = True  # type: ignore[attr-defined]

    root_logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    # ── Console handler (human-readable) ──
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, resolved_level, logging.INFO))
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

    # ── Rotating file handler (JSON or plain) ──
    if log_dir is not None:
        log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / log_file

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, resolved_level, logging.INFO))

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            ))

        root_logger.addHandler(file_handler)
