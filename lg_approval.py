"""Day 3：人工审批 - 危险操作前中断（含去重）"""
import os
import json
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
    """执行数学计算。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误"
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake = {"北京": "晴，22℃", "上海": "多云，25℃"}
    return fake.get(city, f"{city}：暂无数据")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（危险操作，需要人工审批）。"""
    return f"[模拟] 邮件已发送\n收件人：{to}\n主题：{subject}\n正文：{body}"


@tool
def delete_file(path: str) -> str:
    """删除文件（危险操作，需要人工审批）。"""
    return f"[模拟] 已删除：{path}"


safe_tools = [calculator, get_weather]
dangerous_tools = [send_email, delete_file]
all_tools = safe_tools + dangerous_tools
llm_with_tools = llm.bind_tools(all_tools)


# ============ State ============
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============ 加强版 system prompt ============
SYSTEM_PROMPT = """你是一个会使用工具的助手。

规则：
- 需要工具时调用，不需要时直接回答
- 每个工具最多调用一次，得到结果后直接总结给用户
- 不要重复调用相同的工具和参数
"""


# ============ 去重工具 ============
def _tc_key(tc):
    """把一次工具调用转成唯一 key：工具名 + 排序后的参数。"""
    name = tc.get("name") if isinstance(tc, dict) else tc.name
    args = tc.get("args") if isinstance(tc, dict) else tc.args
    return (name, json.dumps(args or {}, sort_keys=True))


def _get_executed_keys(state):
    """收集所有已成功执行过的 (工具名, 参数) 组合。"""
    # 先建立 tool_call_id → key 的映射
    id_to_key = {}
    for m in state["messages"]:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                id_to_key[tc_id] = _tc_key(tc)

    # 找出所有 tool 结果对应的 key
    executed = set()
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            key = id_to_key.get(m.tool_call_id)
            if key:
                executed.add(key)
    return executed


# ============ 节点 ============
def agent_node(state: State) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)

    # 去重：过滤掉已执行过的 tool_calls
    if getattr(response, "tool_calls", None):
        executed = _get_executed_keys(state)
        filtered = [
            tc for tc in response.tool_calls
            if _tc_key(tc) not in executed
        ]

        if not filtered:
            # 全是重复调用 → 强制转成文本回答
            response = AIMessage(
                content=response.content or "操作已完成，请查看上方结果。"
            )
        else:
            response.tool_calls = filtered

    return {"messages": [response]}


def route_after_agent(state: State) -> Literal["safe_tools", "dangerous_tools", "__end__"]:
    last = state["messages"][-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return END

    dangerous_names = {t.name for t in dangerous_tools}
    if any(tc["name"] in dangerous_names for tc in last.tool_calls):
        return "dangerous_tools"
    return "safe_tools"


# ============ 构图 ============
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("safe_tools", ToolNode(safe_tools))
builder.add_node("dangerous_tools", ToolNode(dangerous_tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, {
    "safe_tools": "safe_tools",
    "dangerous_tools": "dangerous_tools",
    END: END,
})
builder.add_edge("safe_tools", "agent")
builder.add_edge("dangerous_tools", "agent")

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_tools"],
)


# ============ 打印 ============
def print_chunks(chunks):
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for node_name, update in chunk.items():
            if isinstance(update, tuple):
                update = update[0] if update else {}
            if not isinstance(update, dict):
                continue
            messages = update.get("messages", [])
            if not isinstance(messages, (list, tuple)):
                continue
            for m in messages:
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            print(f"  [调用] {tc['name']}({tc['args']})")
                    if m.content:
                        print(f"  [回答] {m.content}")
                elif isinstance(m, ToolMessage):
                    print(f"  [返回] {m.content}")


# ============ 审批执行 ============
def run_with_approval(user_input, thread_id="approval"):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }

    print(f"\n你：{user_input}")
    print("-" * 50)

    try:
        print_chunks(graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        ))
    except Exception as e:
        print(f"  [执行异常] {type(e).__name__}: {e}")

    # 最多处理 3 次中断，防止死循环
    for _ in range(3):
        state = graph.get_state(config)
        if not state.next:
            break

        print(f"\n⚠️  需要审批。下一步：{list(state.next)}")
        last_msg = state.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  将要执行：{tc['name']}")
                print(f"  参数：{tc['args']}")

        ans = input("\n是否批准？(y=批准 / n=拒绝)：").strip().lower()

        if ans == "y":
            print("✓ 已批准，继续执行...\n")
            try:
                print_chunks(graph.stream(None, config))
            except Exception as e:
                print(f"  [执行异常] {type(e).__name__}: {e}")
        else:
            print("✗ 已拒绝，任务中止\n")
            break


# ============ 主循环 ============
if __name__ == "__main__":
    print("人工审批 Demo")
    print("示例：'发邮件给 test@example.com 主题 会议 内容 明天10点开会'")
    print("     '北京天气怎么样'（安全工具，不中断）\n")

    thread = "approval-demo"
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        run_with_approval(q, thread)
