"""
日志配置模块。

提供全局 logger 实例，日志同时输出到：
- 文件：agent.log（所有级别）
- 控制台：仅 WARNING 及以上（不干扰正常对话输出）
"""
import logging
from config import LOG_FILE, LOG_LEVEL


def setup_logger(level=None):
    """
    创建并配置全局 logger。

    幂等：多次调用只配置一次，避免重复添加 handler。

    Args:
        level (str|None): 日志级别，None 则使用配置文件的默认值。

    Returns:
        logging.Logger: 已配置的 logger 实例。
    """
    log_level = level or LOG_LEVEL
    logger = logging.getLogger("agent")
    logger.setLevel(log_level)

    # 已配置过则跳过，避免重复输出
    if logger.handlers:
        return logger

    # 统一格式：时间 [级别] 消息
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler：记录所有级别
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler：只显示 WARNING 以上
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)

    return logger


# 全局单例，其他模块直接 from logger import log 使用
log = setup_logger()
