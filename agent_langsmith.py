"""带 LangSmith 可观测性的 Agent"""
import os
import sys
import json
from datetime import datetime

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

_raw_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
client = wrap_openai(_raw_client)

MODEL = "qwen3:8b"


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：非法字符"
    return str(eval(expression, {"__builtins__": {}}, {}))


def get_weather(city: str) -> str:
    fake = {"北京": "晴，22℃", "上海": "多云，25℃"}
    return fake.get(city, f"{city}：暂无数据")


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


TOOL_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。",
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
            "name": "get_weather",
            "description": "查询城市天气。",
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
            "name": "get_time",
            "description": "获取当前时间。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@traceable(run_type="chain", name="react_agent")
def react_once(user_input, max_steps=5):
    messages = [{"role": "user", "content": user_input}]
    seen = set()

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

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

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            key = (name, json.dumps(args, sort_keys=True))
            if key in seen:
                result = "错误：重复调用"
            else:
                seen.add(key)
                fn = TOOL_MAP.get(name)
                if fn:
                    try:
                        result = str(fn(**args))
                    except Exception as e:
                        result = f"错误：{e}"
                else:
                    result = f"未知工具：{name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "（达到最大步数）"


if __name__ == "__main__":
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "")

    print("LangSmith 可观测性 Demo")
    if tracing and api_key:
        print(f"✓ Tracing 已启用，项目：{os.getenv('LANGSMITH_PROJECT')}")
        print(f"  访问 https://smith.langchain.com 查看 trace")
    else:
        print("⚠ 未配置 LANGSMITH_API_KEY，trace 不会上传")
        print("  在 .env 中填入 LANGSMITH_API_KEY 后重启")
    print()

    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        answer = react_once(q)
        print(f"\nAI：{answer}\n")
