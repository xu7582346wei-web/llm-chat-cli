"""安全模块：输入校验、脱敏、限流"""
import re
import time
import logging
from collections import defaultdict
from threading import Lock

log = logging.getLogger("agent")


# ==================== 提示注入检测 ====================
INJECTION_PATTERNS = [
    r"忽略(之前|上面|以上|所有).*(指令|规则|要求|提示)",
    r"(ignore|disregard|forget).{0,20}(previous|above|all).{0,20}(instruction|rule|prompt)",
    r"你的?(系统)?(提示词|prompt|system\s*prompt)",
    r"(泄露|告诉我|输出).{0,10}(系统|提示词|prompt|password|api.?key)",
    r"你现在?是.{0,10}(不是|再也不是)",
    r"<\s*\|.*?\|",
]


def check_injection(text: str) -> tuple[bool, str]:
    """检测疑似提示注入。返回 (是否可疑, 匹配到的模式)"""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, pattern
    return False, ""


# ==================== 敏感信息脱敏 ====================
SENSITIVE_PATTERNS = {
    "api_key": r"sk-[a-zA-Z0-9]{20,}",
    "email": r"[\w\.-]+@[\w\.-]+\.\w+",
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "bank_card": r"\d{16,19}",
}


def mask_sensitive(text: str) -> str:
    """把文本里的敏感信息替换成占位符。"""
    if not isinstance(text, str):
        text = str(text)
    for name, pattern in SENSITIVE_PATTERNS.items():
        text = re.sub(pattern, f"<{name}>", text)
    return text


# ==================== 限流 ====================
class RateLimiter:
    """简单的滑动窗口限流器。"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 86400):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._records = defaultdict(list)
        self._lock = Lock()

    def allow(self, user_id: str) -> bool:
        """检查用户是否还有额度。"""
        now = time.time()
        with self._lock:
            records = self._records[user_id]
            # 清掉过期记录
            cutoff = now - self.window_seconds
            records[:] = [t for t in records if t > cutoff]
            if len(records) >= self.max_requests:
                return False
            records.append(now)
            return True


rate_limiter = RateLimiter(max_requests=200, window_seconds=86400)
