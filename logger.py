"""日志：文件用 JSON 格式（脱敏），控制台用人类可读格式"""
import logging
import json
from datetime import datetime
from config import LOG_FILE, LOG_LEVEL
from security import mask_sensitive


class JsonFormatter(logging.Formatter):
    """JSON 格式 + 敏感信息脱敏。"""

    def format(self, record):
        data = {
            "time": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": mask_sensitive(record.getMessage()),
        }
        for key in ("event", "tool", "args", "elapsed", "model", "tokens", "user"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        return json.dumps(data, ensure_ascii=False)


def setup_logger(level=None):
    log_level = level or LOG_LEVEL
    logger = logging.getLogger("agent")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)

    return logger


log = setup_logger()
