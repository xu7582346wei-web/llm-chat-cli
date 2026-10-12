"""MCP Server - 暴露计算器、天气、知识库工具"""
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer(
    "LLM Agent Tools",
    instructions="提供计算、天气查询、知识库检索工具。",
)


# ==================== 工具 1：计算器 ====================
@mcp.tool()
def calculator(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: Python 数学表达式，如 '(12+8)*3'
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误：{e}"


# ==================== 工具 2：天气 ====================
@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名，如 '北京'
    """
    fake = {
        "北京": "晴，22℃，湿度40%",
        "上海": "多云，25℃，湿度65%",
        "杭州": "小雨，19℃，湿度80%",
        "广州": "雷阵雨，28℃，湿度75%",
    }
    return fake.get(city, f"{city}：暂无数据")


# ==================== 工具 3：当前时间 ====================
@mcp.tool()
def get_time() -> str:
    """获取当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


# ==================== 资源：知识库文档 ====================
@mcp.resource("knowledge://list")
def list_knowledge() -> str:
    """列出知识库中所有文档。"""
    folder = Path("knowledge")
    if not folder.exists():
        return "知识库为空"
    files = [f.name for f in folder.glob("*.md")]
    return "\n".join(files) if files else "知识库为空"


@mcp.resource("knowledge://{filename}")
def read_knowledge(filename: str) -> str:
    """读取知识库中的指定文档。

    Args:
        filename: 文档文件名，如 'faq.md'
    """
    path = Path("knowledge") / filename
    if not path.exists():
        return f"文档不存在：{filename}"
    return path.read_text(encoding="utf-8")


# ==================== 提示模板 ====================
@mcp.prompt()
def summarize_topic(topic: str) -> str:
    """生成一个总结主题的提示模板。

    Args:
        topic: 要总结的主题
    """
    return f"""请从知识库中查找与「{topic}」相关的信息，
然后给出简洁的总结。如果知识库中没有相关信息，请明确说明。"""


# ==================== 启动 ====================
if __name__ == "__main__":
    # stdio 模式：作为子进程运行
    mcp.run()
