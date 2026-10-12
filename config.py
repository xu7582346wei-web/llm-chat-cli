"""
全局配置文件。

所有可调参数集中在这里，方便统一管理。
其他模块通过 `from config import XXX` 引用。

约定：
- 模型名、路径、阈值等硬编码值都放这里
- 运行时可变的状态（如当前会话）不放这里
"""
from pathlib import Path

# ==================== 模型 ====================
# 主对话模型（Ollama 拉取的名称）
MODEL = "qwen3:8b"

# 向量化模型（用于 RAG 的 embedding）
EMBED_MODEL = "nomic-embed-text"

# 采样温度：0 最确定，1 最随机，0.7 是通用默认
TEMPERATURE = 0.7

# ==================== RAG（检索增强） ====================
# 知识库文档所在目录（.md 文件）
KNOWLEDGE_DIR = Path("knowledge")

# 向量检索的最低相似度阈值。
# 低于此值的片段视为不相关，不传给模型。
# 太低 → 噪音多；太高 → 漏掉相关内容。
VECTOR_SCORE_THRESHOLD = 0.5

# 检索返回的 Top-K 片段数量
VECTOR_TOP_K = 3

# ==================== 记忆 ====================
# 长期记忆持久化文件（JSON 格式）
MEMORY_FILE = Path("agent_memory.json")

# 保留最近 N 条消息不参与摘要压缩。
# 超过此数量时，旧消息会被压缩成摘要。
MAX_RECENT_MESSAGES = 10

# ==================== Agent 循环 ====================
# ReAct 单次执行的最大步数（防死循环）
MAX_REACT_STEPS = 5

# Plan-and-Execute 的最大子任务数
MAX_PLAN_STEPS = 5

# ==================== 日志 ====================
# 日志写入文件
LOG_FILE = Path("agent.log")

# 日志级别：DEBUG / INFO / WARNING / ERROR
LOG_LEVEL = "INFO"

# ==================== 会话 ====================
# 保存的会话文件目录
SESSION_DIR = Path("sessions")
