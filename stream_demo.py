import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = "qwen2.5:7b"

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "，写一篇5000字的英语散文"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

print()  # 结尾换行
