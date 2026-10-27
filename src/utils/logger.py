"""Structured logging configuration using Loguru."""

import inspect
import logging
import sys
from pathlib import Path
from loguru import logger

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and redirect them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(
    log_level: str = "INFO",
    json_logs: bool = False,
    log_to_file: bool = True,
) -> None:
    """Configure Loguru structured logging.

    Args:
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: If True, outputs JSON formatted logs to console.
        log_to_file: If True, writes structured logs to files in the `logs/` directory.
    """
    # Remove default Loguru handler
    logger.remove()

    # 1. Console Handler
    if json_logs:
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            colorize=True,
        )

    # 2. File Handlers (Structured JSON logs)
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # General application log (JSON structured, rotated and retained)
        logger.add(
            LOGS_DIR / "app.log",
            level=log_level,
            rotation="10 MB",
            retention="14 days",
            compression="zip",
            encoding="utf-8",
            serialize=True,
            enqueue=True,
        )

        # Separate error log for quick troubleshooting
        logger.add(
            LOGS_DIR / "error.log",
            level="ERROR",
            rotation="5 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            serialize=True,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    # 3. Intercept standard library logging (FastAPI, uvicorn, httpx, etc.)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False


# Initialize default logger setup
setup_logger()

__all__ = ["logger", "setup_logger"]

