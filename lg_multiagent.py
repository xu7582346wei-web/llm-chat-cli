"""Day 5-6：多智能体 - Supervisor 调度（带终止控制）"""
import os
from datetime import datetime
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="qwen3:8b",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ==================== 工具 ====================
@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake = {"北京": "晴，22℃", "上海": "多云，25℃", "杭州": "小雨，19℃"}
    return fake.get(city, f"{city}：暂无数据")


@tool
def calculator(expression: str) -> str:
    """执行数学计算。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误"
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def get_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


# ==================== Worker 定义 ====================
WORKER_PROMPTS = {
    "weather_agent": "你是天气查询专员。只回答天气相关问题，用 get_weather 工具，一句话给出结果。",
    "calc_agent": "你是数学计算专员。用 calculator 工具计算，给出计算过程和结果，一句话。",
    "writer_agent": "你是文案撰写专员。把已有信息整理成一句通顺的话。不要调用工具。",
}

WORKER_TOOLS = {
    "weather_agent": [get_weather],
    "calc_agent": [calculator, get_time],
    "writer_agent": [],
}

worker_llms = {
    name: llm.bind_tools(ts) if ts else llm
    for name, ts in WORKER_TOOLS.items()
}


# ==================== State ====================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_worker: str
    task: str
    visited: list       # 已调用过的 worker 列表
    iterations: int     # 迭代次数


MAX_ITERATIONS = 6


# ==================== Supervisor ====================
SUPERVISOR_PROMPT = """你是团队主管，协调三个成员：

- **weather_agent**：查天气（用 get_weather）
- **calc_agent**：做数学计算（用 calculator）
- **writer_agent**：整理成最终答复（不调工具）

决策规则：
1. 数学计算任务 → 直接给 calc_agent
2. 天气任务 → 直接给 weather_agent
3. 需要对比/整合多源信息 → 先让对应 agent 拿数据，再给 writer_agent 整理
4. **writer_agent 已经完成过 → 立即 FINISH**
5. 已经有足够信息回答用户 → 立即 FINISH
6. 同一个 agent 不要连续调用两次

只输出一个词：weather_agent / calc_agent / writer_agent / FINISH
"""


def supervisor_node(state: State) -> dict:
    task = state.get("task", "")
    visited = state.get("visited", [])
    iterations = state.get("iterations", 0)

    # 硬性终止：已迭代太多次
    if iterations >= MAX_ITERATIONS:
        print(f"  [Supervisor] → FINISH（达到最大迭代 {MAX_ITERATIONS}）")
        return {"next_worker": "FINISH"}

    # 硬性终止：writer 已完成
    if "writer_agent" in visited:
        print(f"  [Supervisor] → FINISH（writer 已完成）")
        return {"next_worker": "FINISH"}

    # 拼装上下文给 supervisor
    context = f"原始任务：{task}\n\n"
    context += f"已调用过的成员：{visited}\n\n"
    context += "已完成的工作：\n"
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.content:
            context += f"- {m.content[:120]}\n"

    resp = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=context),
    ])

    # 解析决策
    decision = "FINISH"
    for name in ["weather_agent", "calc_agent", "writer_agent", "FINISH"]:
        if name in resp.content:
            decision = name
            break

    print(f"  [Supervisor] → {decision}")
    return {"next_worker": decision}


def route_supervisor(state: State) -> Literal["weather_agent", "calc_agent", "writer_agent", "__end__"]:
    nxt = state.get("next_worker", "FINISH")
    if nxt == "FINISH":
        return END
    if nxt in WORKER_PROMPTS:
        return nxt
    return END


# ==================== Worker ====================
def make_worker(name: str):
    system_prompt = WORKER_PROMPTS[name]
    worker_llm = worker_llms[name]
    worker_tool_list = WORKER_TOOLS[name]

    def worker_node(state: State) -> dict:
        print(f"  [{name}] 开始工作")
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
        resp = worker_llm.invoke(messages)
        new_messages = [resp]

        # 如果调了工具，执行并把结果交回模型生成简洁回答
        if hasattr(resp, "tool_calls") and resp.tool_calls:
            tool_map = {t.name: t for t in worker_tool_list}
            for tc in resp.tool_calls:
                fn = tool_map.get(tc["name"])
                if not fn:
                    continue
                try:
                    result = fn.invoke(tc["args"])
                except Exception as e:
                    result = f"错误：{e}"
                print(f"    [工具] {tc['name']} → {str(result)[:60]}")

                follow_up = llm.invoke([
                    SystemMessage(content=system_prompt),
                    *state["messages"],
                    resp,
                    HumanMessage(content=f"工具返回：{result}\n请用一句话给出最终回答。"),
                ])
                # 只保留最终回答，避免中间消息干扰 supervisor
                new_messages = [AIMessage(content=follow_up.content)]
                break

        for m in new_messages:
            if isinstance(m, AIMessage) and m.content:
                print(f"  [{name}] {m.content[:120]}")

        # 更新 visited 和 iterations
        new_visited = list(state.get("visited", []))
        if name not in new_visited:
            new_visited.append(name)

        return {
            "messages": new_messages,
            "visited": new_visited,
            "iterations": state.get("iterations", 0) + 1,
        }

    return worker_node


# ==================== 构图 ====================
builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
for worker_name in WORKER_PROMPTS:
    builder.add_node(worker_name, make_worker(worker_name))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "weather_agent": "weather_agent",
    "calc_agent": "calc_agent",
    "writer_agent": "writer_agent",
    END: END,
})

for worker_name in WORKER_PROMPTS:
    builder.add_edge(worker_name, "supervisor")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ==================== 运行 ====================
def run(user_input, thread_id="multiagent"):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }
    print(f"\n你：{user_input}")
    print("-" * 50)

    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "task": user_input,
            "next_worker": "",
            "visited": [],
            "iterations": 0,
        },
        config,
    )

    print("\n最终回答：")
    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            print(m.content)
            break


if __name__ == "__main__":
    print("多智能体 Demo - Supervisor 调度（带终止控制）")
    print("示例：'北京和上海天气哪个更适合出行'")
    print("     '帮我算 (100+200)*3/5'")
    print("命令：/exit\n")

    thread = "multiagent-demo"
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        run(q, thread)
