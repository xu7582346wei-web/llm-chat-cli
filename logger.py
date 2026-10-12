"""结构化日志。"""
import logging
import sys
from config import LOG_FILE, LOG_LEVEL


def setup_logger(level=None):
    log_level = level or LOG_LEVEL
    logger = logging.getLogger("agent")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)  # 屏幕只显示 WARNING 以上
    logger.addHandler(sh)

    return logger


log = setup_logger()
