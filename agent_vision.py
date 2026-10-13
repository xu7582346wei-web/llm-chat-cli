"""多模态 Agent：通过 MCP 调用视觉工具"""
import os
import sys
import json
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from mcp import Client

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = "qwen3:8b"


class VisionAgent:
    def __init__(self):
        self.mcp_client = None
        self.tools_schema = []
        self.messages = []

    async def connect(self, server):
        self.mcp_client = Client(server)
        await self.mcp_client.__aenter__()

        tools_resp = await self.mcp_client.list_tools()
        for t in tools_resp.tools:
            self.tools_schema.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            })
        print(f"[MCP] 已连接视觉 Server，发现 {len(self.tools_schema)} 个工具")

    async def call_tool(self, name, args):
        result = await self.mcp_client.call_tool(name, args)
        if hasattr(result, "structured_content") and result.structured_content:
            return json.dumps(result.structured_content, ensure_ascii=False)
        return str(result)

    async def chat(self, user_input, max_steps=5):
        self.messages.append({"role": "user", "content": user_input})

        for step in range(max_steps):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self.tools_schema,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            self.messages.append({
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

                print(f"  [视觉工具] {name}({args})")
                result = await self.call_tool(name, args)
                print(f"  [返回] {str(result)[:150]}...")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return "（达到最大步数）"

    async def close(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)


async def main():
    from mcp_vision_server import mcp

    agent = VisionAgent()
    await agent.connect(mcp)

    print("\n多模态 Agent 已就绪")
    print("试试：'看看 test_image.png 里有什么内容'")
    print("命令：/exit\n")

    try:
        while True:
            try:
                q = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q in ("/exit", ""):
                break
            answer = await agent.chat(q)
            print(f"\nAI：{answer}\n")
    finally:
        try:
            await agent.close()
        except (asyncio.CancelledError, KeyboardInterrupt, Exception):
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
