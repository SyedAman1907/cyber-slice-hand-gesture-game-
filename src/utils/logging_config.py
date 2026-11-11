"""
Logging Configuration Module.
Provides centralized, thread-safe, and formatted logging across all system components.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "CV_Gesture_Game", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized logger instance.
    
    Args:
        name: Name of the logger or module.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Fall back to console only if file access fails

    return logger
