"""Agent 配置。所有可调参数集中在这里。"""
from pathlib import Path

# ============ 模型 ============
MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.7

# ============ RAG ============
KNOWLEDGE_DIR = Path("knowledge")
VECTOR_SCORE_THRESHOLD = 0.5
VECTOR_TOP_K = 3

# ============ 记忆 ============
MEMORY_FILE = Path("agent_memory.json")
MAX_RECENT_MESSAGES = 10

# ============ Agent ============
MAX_REACT_STEPS = 5
MAX_PLAN_STEPS = 5

# ============ 日志 ============
LOG_FILE = Path("agent.log")
LOG_LEVEL = "INFO"

# ============ 会话 ============
SESSION_DIR = Path("sessions")
