"""
LLM Agent 主程序。

架构总览：
    ┌─────────────────────────────────────────┐
    │ 用户输入                                 │
    │   ↓                                     │
    │ 判断复杂度 → ReAct / Plan-and-Execute    │
    │   ↓                                     │
    │ 循环：思考 → 调工具 → 观察 → 再思考       │
    │   ↓                                     │
    │ 输出最终回答                             │
    └─────────────────────────────────────────┘

核心能力：
    - 工具调用（Function Calling + 参数校验 + 错误处理）
    - 短期记忆（messages 数组）+ 长期记忆（JSON 持久化）
    - 任务规划（Plan-and-Execute）
    - RAG（向量检索 + 阈值过滤）
    - 会话保存 / 加载

运行方式：
    python agent.py                     # 交互模式
    python agent.py -q "问题"            # 单次提问
    python agent.py --help              # 查看参数
"""
import os
import sys
import re
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, ValidationError

from config import (
    MODEL, EMBED_MODEL, TEMPERATURE,
    KNOWLEDGE_DIR, VECTOR_SCORE_THRESHOLD, VECTOR_TOP_K,
    MEMORY_FILE, MAX_RECENT_MESSAGES,
    MAX_REACT_STEPS, MAX_PLAN_STEPS,
    SESSION_DIR,
)
from cli import parse_args
from logger import log

# 终端编码容错，防止中文输入输出乱码（WSL 常见问题）
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# OpenAI 兼容客户端（指向 Ollama / DeepSeek / 通义等）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 运行时可变状态（可被命令行参数覆盖）
RUNTIME_MODEL = MODEL
USE_MEMORY = True


# ==================== System Prompts ====================
# 主 Agent 的行为准则。核心是"先思考再行动"和"不编造"。
SYSTEM_PROMPT = """/no_think
你是一个会使用工具的 Agent。**必须优先使用工具，不要凭训练记忆回答**。
每次行动前，先用一句话说明你的思考。
然后决定：
- 需要工具 → 通过标准 function call 调用
- 已有足够信息 → 直接回答

规则：
- 不要编造工具返回的数据
- 公司政策、流程、常见问题 → 用 search_knowledge 查
- 实时信息（天气、时间）→ 用对应工具
- 计算必须用 calculator
- 用户问当前时间/日期 → 必须用 get_time，不要凭记忆回答
- 调用工具必须用 function call 机制，不要在回答文本里写 {"name": ...}
"""

# 任务规划器：把复杂任务拆成子任务
PLANNER_PROMPT = """你是一个任务规划器。
把用户任务拆解成 2-5 个可独立执行的子任务。

输出格式（严格 JSON）：
{"plan": [{"step": 1, "task": "..."}, {"step": 2, "task": "..."}]}
"""

# 结果汇总器：综合各步骤结果给出最终答复
SUMMARIZER_PROMPT = "你是结果汇总器。综合各步骤结果，直接回答用户问题，简洁明了。"


# ==================== 长期记忆 ====================
def load_memory():
    """
    从磁盘加载长期记忆。

    Returns:
        dict: {"facts": [str, ...]}，文件不存在或损坏时返回空结构。
    """
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # 文件损坏时静默重置，避免程序崩溃
            pass
    return {"facts": []}


def save_memory(mem):
    """把长期记忆写回磁盘（UTF-8，保留中文）。"""
    MEMORY_FILE.write_text(
        json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def remember_fact(fact: str) -> str:
    """
    工具：记住一条事实。去重后持久化。

    Args:
        fact: 要记住的内容，例如"用户喜欢喝美式咖啡"。

    Returns:
        给模型看的结果字符串。
    """
    mem = load_memory()
    if fact not in mem["facts"]:
        mem["facts"].append(fact)
        save_memory(mem)
    return f"已记住：{fact}"


def recall_facts() -> str:
    """工具：回忆所有已记住的事实。"""
    mem = load_memory()
    if not mem["facts"]:
        return "目前没有记住任何事实"
    return "记住的事实：\n" + "\n".join(f"- {f}" for f in mem["facts"])


# ==================== 会话保存 ====================
def save_session(messages, path=None):
    """
    把对话历史保存为 JSON 文件。

    Args:
        messages: 消息列表（含 system/user/assistant/tool）
        path: 保存路径；None 则自动生成带时间戳的文件名。

    Returns:
        Path: 实际保存的文件路径。
    """
    SESSION_DIR.mkdir(exist_ok=True)
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SESSION_DIR / f"session_{ts}.json"
    else:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)

    # 只保留可序列化的字段，避免写入 OpenAI 对象
    clean = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        item = {"role": m.get("role"), "content": m.get("content", "")}
        if m.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    },
                }
                for tc in m["tool_calls"]
            ]
        if m.get("tool_call_id"):
            item["tool_call_id"] = m["tool_call_id"]
        clean.append(item)

    data = {
        "saved_at": datetime.now().isoformat(),
        "model": RUNTIME_MODEL,
        "messages": clean,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_session(path):
    """
    从 JSON 文件加载对话历史。

    Args:
        path: 会话文件路径。

    Returns:
        list: 消息列表。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"会话文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("messages", [])


def list_sessions():
    """列出 sessions/ 下所有会话文件，按时间倒序。"""
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("*.json"), reverse=True)


# ==================== RAG ====================
def embed(texts):
    """
    批量文本向量化。

    Args:
        texts: 字符串列表。

    Returns:
        list[list[float]]: 每个文本对应的向量。
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def chunk_text(text, chunk_size=500):
    """
    按 Markdown 结构切分文档。

    策略：
    1. 遇到一级标题 # → 记为新语境前缀
    2. 遇到二级标题 ## → 开一个新 chunk
    3. 每个 chunk 带上所属的一级标题（保留语境）
    4. 超长 chunk 再按长度切，带 50 字符重叠

    好处：每个 chunk 能独立被理解，检索精度更高。

    Args:
        text: 原始文档内容。
        chunk_size: 单 chunk 最大字符数。

    Returns:
        list[str]: 切分后的片段。
    """
    lines = text.split("\n")
    chunks = []
    current_h1 = ""      # 当前一级标题
    current_chunk = []   # 当前累积的行

    def flush():
        """把当前累积的内容提交为一个 chunk。"""
        if not current_chunk:
            return
        content = "\n".join(current_chunk).strip()
        if not content:
            return
        # 拼接一级标题作为语境
        full = f"{current_h1}\n{content}" if current_h1 else content
        if len(full) <= chunk_size:
            chunks.append(full)
        else:
            # 超长则滑动窗口切分，保留重叠避免语义断裂
            start = 0
            while start < len(full):
                chunks.append(full[start:start + chunk_size])
                start += chunk_size - 50

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush()
            current_chunk = []
            current_h1 = stripped
        elif stripped.startswith("## "):
            flush()
            current_chunk = [line]
        else:
            current_chunk.append(line)
    flush()
    return [c for c in chunks if c.strip()]


def load_documents(folder):
    """
    加载目录下所有 .md 文档并切分。

    Args:
        folder: 目录路径。

    Returns:
        list[dict]: 每个元素 {"text": str, "source": str, "chunk_id": int}
    """
    folder = Path(folder)
    chunks = []
    if not folder.exists():
        return chunks
    for path in folder.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for i, c in enumerate(chunk_text(text)):
            chunks.append({
                "text": f"[文件: {path.name}]\n{c}",
                "source": path.name,
                "chunk_id": i,
            })
    return chunks


class VectorStore:
    """
    内存向量库。

    存储 chunk 和对应的归一化向量，支持余弦相似度检索。
    向量归一化后，余弦相似度 = 点积，加速计算。
    """

    def __init__(self):
        self.chunks = []
        self.vectors = None

    def add(self, chunks):
        """批量添加 chunk 并计算向量。"""
        self.chunks = chunks
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        log.info(f"索引 {len(texts)} 个片段")
        vectors = embed(texts)
        self.vectors = np.array(vectors, dtype=np.float32)
        # L2 归一化：||v|| = 1，方便用点积算余弦相似度
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-10)

    def search(self, query, top_k=VECTOR_TOP_K, threshold=VECTOR_SCORE_THRESHOLD):
        """
        检索最相似的 top_k 个 chunk。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            threshold: 相似度阈值，低于此值的丢弃。

        Returns:
            list[dict]: 按相似度降序的 chunk，附带 "score" 字段。
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []
        # 查询向量也要归一化
        q_vec = np.array(embed([query])[0], dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        # 归一化后点积 = 余弦相似度
        scores = self.vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            score = float(scores[i])
            if score < threshold:
                break  # 已降序，后面更低，直接停
            results.append({**self.chunks[i], "score": score})
        return results


# 全局向量库（在 main() 中初始化）
STORE = None


def search_knowledge(query: str) -> str:
    """
    工具：从知识库检索相关信息。

    Args:
        query: 检索关键词或问题。

    Returns:
        拼接后的检索结果文本，或"未找到"提示。
    """
    if STORE is None:
        return "知识库未初始化"
    hits = STORE.search(query)
    if not hits:
        return "知识库中没有找到相关信息"
    return "\n\n---\n\n".join(
        f"[来源：{h['source']}，相似度 {h['score']:.3f}]\n{h['text']}"
        for h in hits
    )


# ==================== 参数 Schema ====================
# 用 Pydantic 校验模型传来的工具参数，防止非法输入。
# 校验失败会返回错误信息给模型，让它自己修正。

class CalculatorArgs(BaseModel):
    """calculator 工具的参数。"""
    expression: str

    @field_validator("expression")
    @classmethod
    def check(cls, v):
        """白名单校验：只允许数字和基本运算符。"""
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in v):
            raise ValueError("只能包含数字和 +-*/(). 符号")
        return v


class WeatherArgs(BaseModel):
    """get_weather 工具的参数。"""
    city: str


class RememberArgs(BaseModel):
    """remember_fact 工具的参数。"""
    fact: str


class SearchArgs(BaseModel):
    """search_knowledge 工具的参数。"""
    query: str


# ==================== 工具实现 ====================
def calculator(expression: str) -> str:
    """
    工具：安全数学计算。

    用 eval 但清空 __builtins__，只允许纯数学表达式。
    """
    return str(eval(expression, {"__builtins__": {}}, {}))


def get_time() -> str:
    """工具：获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_weather(city: str) -> str:
    """工具：查询城市天气（此处为假数据，可替换为真实 API）。"""
    fake = {
        "北京": "晴，22℃，湿度40%",
        "上海": "多云，25℃，湿度65%",
        "杭州": "小雨，19℃，湿度80%",
        "广州": "雷阵雨，28℃，湿度75%",
    }
    return fake.get(city, f"{city}：暂无数据")


# 工具名 → (函数, 参数 Schema) 的映射
TOOL_MAP = {
    "calculator": (calculator, CalculatorArgs),
    "get_time": (get_time, None),
    "get_weather": (get_weather, WeatherArgs),
    "remember_fact": (remember_fact, RememberArgs),
    "recall_facts": (recall_facts, None),
    "search_knowledge": (search_knowledge, SearchArgs),
}

# 传给模型的工具定义（JSON Schema 格式）。
# description 决定模型何时选择该工具，务必写清楚使用场景。
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。当用户需要做算术、算百分比、算价格、算折扣时必须调用，不要心算。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期、时间、星期几。当用户问'现在几点'、'今天几号'、'今天星期几'时必须调用，不要凭训练记忆回答。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气（温度、天气状况）。当用户问某地天气、温度、是否下雨时必须调用。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "当用户说出重要信息、偏好、长期有效的事实时调用。",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "用户问'你记得什么'时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "从公司内部知识库检索信息。当用户问公司政策、流程、年假、报销、请假等常见问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"}
                },
                "required": ["query"],
            },
        },
    },
]


def safe_run(name, args):
    """
    安全执行工具：参数校验 + 异常捕获 + 结果截断。

    任何错误都转为字符串返回（不抛异常），
    让模型看到错误后自行决定下一步。

    Args:
        name: 工具名。
        args: 参数字典。

    Returns:
        工具结果或错误信息的字符串。
    """
    if name not in TOOL_MAP:
        return f"错误：工具 '{name}' 不存在"
    func, schema = TOOL_MAP[name]

    # 参数校验（有 schema 的工具才校验）
    if schema:
        try:
            args = schema(**args).model_dump()
        except ValidationError as e:
            return f"参数校验失败：{e.errors()}"

    # 执行工具，捕获所有异常
    try:
        result = func(**args)
    except Exception as e:
        log.exception(f"工具 {name} 执行失败")
        return f"工具执行异常：{type(e).__name__}: {e}"

    # 结果截断，防止塞爆上下文
    s = str(result)
    return s[:2000] + "...(截断)" if len(s) > 2000 else s


# ==================== ReAct 执行循环 ====================
def react_once(task, max_steps=None, verbose=True, messages=None, on_event=None):
    """
    单次 ReAct 执行：思考 → 行动 → 观察 → 再思考，直到有答案。

    Args:
        task: 任务描述（用户输入或子任务）。
        max_steps: 最大循环步数，防死循环。
        verbose: 是否打印过程到终端。
        messages: 已有对话历史；None 则新建。
        on_event: 事件回调，签名 (kind, data)，用于 Web UI 推送。

    Returns:
        str: 最终回答文本。
    """
    max_steps = max_steps or MAX_REACT_STEPS

    # 无历史时，新建上下文
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

    # 记录已调用的 (工具名, 参数)，防止重复调用
    seen = set()

    for step in range(1, max_steps + 1):
        # ---------- 1. 调用模型 ----------
        try:
            resp = client.chat.completions.create(
                model=RUNTIME_MODEL,
                messages=messages,
                tools=tools,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            log.exception("调用模型失败")
            return f"调用模型失败：{e}"

        msg = resp.choices[0].message

        # ---------- 2. 没有工具调用 → 得到最终答案 ----------
        if not msg.tool_calls:
            return msg.content

        # ---------- 3. 有工具调用 → 加入历史 ----------
        # 手动构造 tool_calls，避免 model_dump 带入多余字段导致模型混乱
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # ---------- 4. 逐个执行工具 ----------
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # 重复调用检测
            key = (name, json.dumps(args, sort_keys=True))
            if key in seen:
                result = "错误：重复调用"
            else:
                seen.add(key)
                result = safe_run(name, args)

            if verbose:
                preview = str(result)[:80].replace("\n", " ")
                print(f"  [工具] {name}({args}) → {preview}...")

            log.info(f"tool_call name={name} args={args}")

            # 推送给 Web UI
            if on_event:
                on_event("tool_call", {
                    "name": name,
                    "args": args,
                    "result": str(result)[:500],
                })

            # 工具结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "（达到最大步数）"


# ==================== Plan-and-Execute ====================
def make_plan(user_input):
    """
    让模型把复杂任务拆解成子任务列表。

    Args:
        user_input: 用户原始问题。

    Returns:
        list[dict]: [{"step": int, "task": str}, ...]，失败返回空列表。
    """
    try:
        resp = client.chat.completions.create(
            model=RUNTIME_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
        )
        plan = json.loads(resp.choices[0].message.content)
        return plan.get("plan", [])[:MAX_PLAN_STEPS]
    except Exception as e:
        log.warning(f"规划失败：{e}")
        return []


def summarize(user_input, results):
    """把所有子任务结果汇总成最终答复。"""
    text = f"原始问题：{user_input}\n\n各步骤结果：\n"
    for r in results:
        text += f"[Step {r['step']}] {r['task']}\n结果：{r['result']}\n\n"
    resp = client.chat.completions.create(
        model=RUNTIME_MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content


def plan_execute(user_input, verbose=True, on_event=None):
    """
    Plan-and-Execute 流程：
    1. 规划器拆解任务
    2. 逐个用 ReAct 执行子任务
    3. 汇总结果

    规划失败时降级为单次 ReAct。
    """
    if verbose:
        print(f"\n[规划中] ...")
    plan = make_plan(user_input)
    if not plan:
        if verbose:
            print("[降级] 直接 ReAct")
        return react_once(user_input, verbose=verbose, on_event=on_event)

    if verbose:
        print(f"[计划] 共 {len(plan)} 步：")
        for item in plan:
            print(f"  {item.get('step', '?')}. {item.get('task', '')}")

    if on_event:
        on_event("plan", {"plan": plan})

    # 逐个执行子任务
    results = []
    for item in plan:
        task = item.get("task", "")
        if verbose:
            print(f"\n[执行 Step {item.get('step', '?')}] {task}")
        answer = react_once(task, verbose=verbose, on_event=on_event)
        results.append({
            "step": item.get("step", "?"),
            "task": task,
            "result": answer,
        })
        if verbose:
            print(f"  → {answer}")

    if verbose:
        print(f"\n[汇总] ...")
    return summarize(user_input, results)


# ==================== Agent ====================
class Agent:
    """
    会话级 Agent。

    维护对话历史（messages），提供 chat/clear/save/load 方法。
    短期记忆存在 self.messages 里，进程结束即丢失。
    长期记忆通过 load_memory/save_memory 持久化。
    """

    def __init__(self):
        self._reset_messages()

    def _reset_messages(self):
        """重置对话历史，并把长期记忆注入 system prompt。"""
        long_term = ""
        if USE_MEMORY:
            mem = load_memory()
            if mem["facts"]:
                long_term = "\n\n[长期记忆]\n" + "\n".join(f"- {f}" for f in mem["facts"])
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT + long_term}]

    def chat(self, user_input, verbose=True, on_event=None):
        """
        处理一轮对话。

        根据输入长度和关键词判断是否需要规划：
        - 简单问题 → ReAct
        - 复杂任务 → Plan-and-Execute

        Args:
            user_input: 用户输入。
            verbose: 是否打印过程。
            on_event: 事件回调（Web UI 用）。

        Returns:
            str: AI 回答。
        """
        # ---- 安全：输入注入检测 ----
        from security import check_injection, rate_limiter
        import logging
        sec_log = logging.getLogger("agent")

        suspicious, pattern = check_injection(user_input)
        if suspicious:
            sec_log.warning(
                f"检测到疑似提示注入：{pattern}",
                extra={"event": "injection_blocked", "user": "default"},
            )
            return "抱歉，你的输入包含可疑指令，我无法处理。"

        # ---- 安全：限流 ----
        if not rate_limiter.allow("default"):
            return "今日请求次数已达上限，请明天再试。"

        self.messages.append({"role": "user", "content": user_input})

        # 启发式判断：长输入或含特定关键词则用规划
        need_plan = (
            len(user_input) > 30
            or "然后" in user_input
            or "并且" in user_input
            or "对比" in user_input
        )

        if need_plan:
            answer = plan_execute(user_input, verbose=verbose, on_event=on_event)
        else:
            answer = react_once(
                user_input,
                verbose=verbose,
                messages=self.messages,
                on_event=on_event,
            )

        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def clear(self):
        """清空对话历史（保留长期记忆）。"""
        self._reset_messages()

    def show_history(self):
        """打印当前对话历史。"""
        for m in self.messages:
            if isinstance(m, dict):
                role = m.get("role")
                content = str(m.get("content", "") or "")
                if len(content) > 150:
                    content = content[:150] + "..."
                print(f"[{role}] {content}")

    def save(self, path=None):
        """保存会话。"""
        return save_session(self.messages, path)

    def load(self, path):
        """加载会话（替换当前 messages）。"""
        self.messages = load_session(path)


# ==================== 主程序 ====================
def main():
    """命令行入口。"""
    # 这些是模块级可变状态，需要在函数内声明
    global RUNTIME_MODEL, USE_MEMORY, STORE

    args = parse_args()

    # 命令行参数覆盖默认值
    if args.model:
        RUNTIME_MODEL = args.model
    if args.no_memory:
        USE_MEMORY = False

    # 初始化知识库（除非 --no-rag）
    if not args.no_rag:
        STORE = VectorStore()
        chunks = load_documents(KNOWLEDGE_DIR)
        if chunks:
            print(f"[知识库] 加载 {len(chunks)} 个片段")
            STORE.add(chunks)
        else:
            print("[知识库] 未找到文档，跳过")

    agent = Agent()

    # 单次提问模式
    if args.query:
        answer = agent.chat(args.query)
        print(f"\nAI：{answer}")
        return

    # 交互模式
    print(f"\nAgent v4 - 模型：{RUNTIME_MODEL}")
    print("命令：/clear /history /save [文件名] /load <文件名> /sessions /exit")
    print("=" * 60)

    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # 内置命令处理
        if q == "/exit":
            break

        if q == "/clear":
            agent.clear()
            print("历史已清空")
            continue

        if q == "/history":
            agent.show_history()
            continue

        if q == "/save":
            path = agent.save()
            print(f"会话已保存：{path}")
            continue

        if q.startswith("/save "):
            name = q[6:].strip()
            path = SESSION_DIR / name
            if not path.suffix:
                path = path.with_suffix(".json")
            saved = agent.save(path)
            print(f"会话已保存：{saved}")
            continue

        if q.startswith("/load "):
            name = q[6:].strip()
            path = SESSION_DIR / name
            if not path.exists():
                path = SESSION_DIR / f"{name}.json"
            try:
                agent.load(path)
                print(f"会话已加载：{path}")
                print(f"共 {len(agent.messages)} 条消息")
            except Exception as e:
                print(f"加载失败：{e}")
            continue

        if q == "/sessions":
            sessions = list_sessions()
            if not sessions:
                print("暂无保存的会话")
            else:
                print(f"共 {len(sessions)} 个会话：")
                for s in sessions[:20]:
                    size = s.stat().st_size
                    print(f"  - {s.name}  ({size} 字节)")
            continue

        if not q:
            continue

        # 正常对话
        answer = agent.chat(q)
        print(f"\nAI：{answer}")

    # 退出时可选保存
    if args.save_session:
        path = agent.save()
        print(f"会话已保存：{path}")


if __name__ == "__main__":
    main()
