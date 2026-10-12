"""Day 2：LangGraph + Checkpointer（多轮对话）+ 流式输出"""
import os
from datetime import datetime
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="qwen3:8b",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============ 工具 ============
@tool
def calculator(expression: str) -> str:
    """执行数学计算。参数是 Python 数学表达式，如 '12+8'。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：非法字符"
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def get_time() -> str:
    """获取当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake = {"北京": "晴，22℃", "上海": "多云，25℃", "杭州": "小雨，19℃"}
    return fake.get(city, f"{city}：暂无数据")


tools = [calculator, get_time, get_weather]
llm_with_tools = llm.bind_tools(tools)


# ============ State ============
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============ 节点 ============
SYSTEM_PROMPT = "你是一个会使用工具的助手。需要工具时调用，否则直接回答。"


def agent_node(state: State) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    return {"messages": [llm_with_tools.invoke(messages)]}


def should_continue(state: State) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ============ 构图 ============
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

# 关键：加上 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ============ 对话 ============
def chat(user_input, thread_id="default"):
    """执行一轮，流式打印过程。"""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }

    print(f"\n你：{user_input}")
    print("-" * 50)

    # stream 模式：每执行完一个节点，yield 一次
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    ):
        for node_name, update in chunk.items():
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            print(f"  [调用] {tc['name']}({tc['args']})")
                    if m.content:
                        print(f"  [回答] {m.content}")
                elif isinstance(m, ToolMessage):
                    print(f"  [返回] {m.content}")


def show_history(thread_id="default"):
    """查看某个 thread 的完整历史。"""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    print(f"\n[Thread: {thread_id}] 历史消息：")
    for m in state.values.get("messages", []):
        name = type(m).__name__
        content = str(getattr(m, "content", ""))[:80]
        print(f"  [{name}] {content}")


# ============ 主循环 ============
if __name__ == "__main__":
    print("LangGraph + Checkpointer。命令：/history /switch <id> /threads /exit")
    thread = "default"

    while True:
        try:
            q = input(f"\n你({thread})：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q == "/exit":
            break
        if q == "/history":
            show_history(thread)
            continue
        if q.startswith("/switch "):
            thread = q[8:].strip()
            print(f"已切换到 thread：{thread}")
            continue
        if not q:
            continue

        chat(q, thread_id=thread)
