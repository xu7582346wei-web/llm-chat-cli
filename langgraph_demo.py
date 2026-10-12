"""最小 LangGraph 示例：一个会思考、会调工具的 Agent。"""
import os
import json
from typing import Annotated, TypedDict
from operator import add

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ==================== 1. LLM ====================
# langchain-openai 也兼容 OpenAI 接口，指向本地 Ollama
llm = ChatOpenAI(
    model="qwen3:8b",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)


# ==================== 2. 定义工具 ====================
@tool
def calculator(expression: str) -> str:
    """执行数学计算。参数是 Python 数学表达式，如 '12+8'。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_time() -> str:
    """获取当前日期时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake = {"北京": "晴，22℃", "上海": "多云，25℃", "杭州": "小雨，19℃"}
    return fake.get(city, f"{city}：暂无数据")


tools = [calculator, get_time, get_weather]
llm_with_tools = llm.bind_tools(tools)


# ==================== 3. State ====================
class State(TypedDict):
    """Agent 的共享状态。
    
    messages 用 add_messages 装饰，表示每次更新是「追加」而非「替换」。
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ==================== 4. 节点 ====================
SYSTEM_PROMPT = "你是一个会使用工具的助手。需要计算、查时间、查天气时调用工具，否则直接回答。"


def call_model(state: State) -> dict:
    """节点：调用模型。"""
    messages = state["messages"]
    # 首次调用时注入 system prompt
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 预置的 ToolNode 会自动执行 tool_calls
tool_node = ToolNode(tools)


# ==================== 5. 条件边 ====================
def should_continue(state: State) -> str:
    """根据最后一条消息判断下一步。
    
    - 有 tool_calls → 去执行工具
    - 无 → 结束
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ==================== 6. 构建图 ====================
graph_builder = StateGraph(State)

graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

# 入口
graph_builder.set_entry_point("agent")

# 条件边：agent 之后判断走 tools 还是结束
graph_builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END,
})

# tools 执行完回到 agent
graph_builder.add_edge("tools", "agent")

# 编译
graph = graph_builder.compile()


# ==================== 7. 运行 ====================
def chat(user_input: str):
    """执行一轮对话，打印过程。"""
    print(f"\n你：{user_input}")
    print("-" * 50)

    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 20},
    )

    # 打印中间过程
    for msg in final_state["messages"]:
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [调用] {tc['name']}({tc['args']})")
            if msg.content:
                print(f"  [回答] {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"  [返回] {msg.content}")


if __name__ == "__main__":
    print("LangGraph Agent 演示。输入 /exit 退出\n")
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        chat(q)
