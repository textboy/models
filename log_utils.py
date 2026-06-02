"""Shared logging setup for gen_prompt, z_image, and batch.

Usage:
    from log_utils import setup_logger
    log = setup_logger("gen_prompt")
    log.info("Hello")
    log.error("Something went wrong")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logger(name):
    """Create a logger that writes to both console and logs/{name}_yyyymmdd.log.

    Args:
        name: short name used in the log filename (e.g. "gen_prompt", "batch")

    Returns:
        logging.Logger instance
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"{name}_{today}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # File handler — full detail
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Console handler — info and above, no timestamps
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger
