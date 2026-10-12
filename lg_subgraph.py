"""Day 4：子图 - 把 Plan-and-Execute 封装成子图"""
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


# ==================== 工具 ====================
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
    fake = {"北京": "晴，22℃", "上海": "多云，25℃", "杭州": "小雨，19℃"}
    return fake.get(city, f"{city}：暂无数据")


@tool
def get_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


tools = [calculator, get_weather, get_time]
llm_with_tools = llm.bind_tools(tools)


# ==================== 子图：Plan-and-Execute ====================
class PlanState(TypedDict):
    """子图状态。"""
    task: str                    # 原始任务
    plan: list                   # 规划出的子任务列表
    current_step: int            # 当前执行到第几步
    results: list                # 已完成子任务的结果
    final_answer: str            # 汇总后的最终答案


PLANNER_PROMPT = """你是任务规划器。
把用户任务拆解成 2-4 个可独立执行的子任务。

输出严格 JSON：
{"plan": ["子任务1", "子任务2", "子任务3"]}
"""

EXECUTOR_PROMPT = """你是一个执行子任务的助手。
只完成当前这一个子任务，用一句话给出结果，不要啰嗦。
"""

SUMMARIZER_PROMPT = """你是结果汇总器。
综合以下各步骤结果，直接回答用户的原始问题，简洁明了。"""


def plan_node(state: PlanState) -> dict:
    """节点 1：拆解任务。"""
    resp = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=state["task"]),
    ])
    try:
        content = resp.content.strip()
        # 去掉可能的 ```json 包装
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(content)
        plan = data.get("plan", [])
    except Exception as e:
        print(f"  [规划失败] {e}")
        plan = [state["task"]]  # 降级：整个任务一步完成

    print(f"  [规划] {len(plan)} 步：{plan}")
    return {"plan": plan, "current_step": 0, "results": []}


def should_execute_more(state: PlanState) -> Literal["execute", "summarize"]:
    """条件边：还有子任务就继续，否则汇总。"""
    if state["current_step"] >= len(state["plan"]):
        return "summarize"
    return "execute"


def execute_node(state: PlanState) -> dict:
    """节点 2：执行当前子任务。"""
    step = state["current_step"]
    task = state["plan"][step]
    print(f"  [执行 step {step + 1}] {task}")

    # 每个子任务独立调用 ReAct（简单版：直接用模型+工具）
    messages = [
        SystemMessage(content=EXECUTOR_PROMPT),
        HumanMessage(content=task),
    ]

    # 简单 ReAct 循环
    for _ in range(3):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)
        if not (hasattr(resp, "tool_calls") and resp.tool_calls):
            break
        for tc in resp.tool_calls:
            tool_map = {t.name: t for t in tools}
            fn = tool_map.get(tc["name"])
            if fn:
                try:
                    result = fn.invoke(tc["args"])
                except Exception as e:
                    result = f"错误：{e}"
                print(f"    [工具] {tc['name']} → {str(result)[:60]}")
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

    # 提取最终文本
    final = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            final = m.content
            break

    return {
        "current_step": step + 1,
        "results": state["results"] + [{
            "step": step + 1,
            "task": task,
            "result": final,
        }],
    }


def summarize_node(state: PlanState) -> dict:
    """节点 3：汇总所有子任务结果。"""
    print(f"  [汇总] 综合 {len(state['results'])} 个结果")
    text = f"原始任务：{state['task']}\n\n各步骤结果：\n"
    for r in state["results"]:
        text += f"[{r['step']}] {r['task']}\n→ {r['result']}\n\n"

    resp = llm.invoke([
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=text),
    ])
    return {"final_answer": resp.content}


# 构建子图
plan_builder = StateGraph(PlanState)
plan_builder.add_node("plan", plan_node)
plan_builder.add_node("execute", execute_node)
plan_builder.add_node("summarize", summarize_node)

plan_builder.add_edge(START, "plan")
plan_builder.add_conditional_edges("plan", should_execute_more, {
    "execute": "execute",
    "summarize": "summarize",
})
plan_builder.add_conditional_edges("execute", should_execute_more, {
    "execute": "execute",
    "summarize": "summarize",
})
plan_builder.add_edge("summarize", END)

plan_subgraph = plan_builder.compile()


# ==================== 主图 ====================
class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """你是一个会使用工具的助手。

规则：
- 简单问题直接回答或用工具
- 复杂任务（多步骤、需要对比、需要整合多个信息源）→ 使用 plan_execute 工具
- 每个工具最多调用一次
"""


@tool
def plan_execute(task: str) -> str:
    """处理复杂任务：自动拆解为多个子任务并逐个执行。
    
    适用于：需要多步推理、对比分析、整合多个信息源的任务。
    
    Args:
        task: 完整的任务描述
    """
    print(f"\n[进入子图] {task}")
    result = plan_subgraph.invoke({
        "task": task,
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": "",
    })
    return result["final_answer"]


main_tools = tools + [plan_execute]
llm_main = llm.bind_tools(main_tools)


def main_agent_node(state: MainState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    return {"messages": [llm_main.invoke(messages)]}


def main_should_continue(state: MainState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


main_builder = StateGraph(MainState)
main_builder.add_node("agent", main_agent_node)
main_builder.add_node("tools", ToolNode(main_tools))
main_builder.add_edge(START, "agent")
main_builder.add_conditional_edges("agent", main_should_continue, {
    "tools": "tools",
    END: END,
})
main_builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = main_builder.compile(checkpointer=memory)


# ==================== 运行 ====================
def chat(user_input, thread_id="subgraph"):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }
    print(f"\n你：{user_input}")
    print("-" * 50)

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    ):
        if not isinstance(chunk, dict):
            continue
        for node, update in chunk.items():
            if isinstance(update, tuple):
                update = update[0] if update else {}
            if not isinstance(update, dict):
                continue
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            print(f"  [调用] {tc['name']}({tc['args']})")
                    if m.content:
                        print(f"  [回答] {m.content}")
                elif isinstance(m, ToolMessage):
                    preview = str(m.content)[:100].replace("\n", " ")
                    print(f"  [返回] {preview}")


if __name__ == "__main__":
    print("子图 Demo - 主图 + Plan-and-Execute 子图")
    print("命令：/exit\n")

    thread = "subgraph-demo"
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        chat(q, thread)
