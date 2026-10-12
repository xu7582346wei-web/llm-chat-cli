import os
import sys
import time
import logging
from datetime import datetime

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from dotenv import load_dotenv

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("chat.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("chat")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=60,
)

MODEL = "qwen2.5:7b"

BASE_SYSTEM = """你是一个有用、简洁的助手。
根据用户的问题调整回答方式：
- 问代码 -> 给代码
- 问创作 -> 直接写，不要套代码
- 问事实 -> 直接回答
不要无脑套用某一种格式。
"""


def build_system_prompt():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
    return f"{BASE_SYSTEM}\n当前时间：{now}"


state = {
    "temperature": 0.7,
    "max_tokens": None,
    "stream": True,
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}


def with_retry(func, retries=3, base_delay=1.0):
    for attempt in range(1, retries + 1):
        try:
            return func()
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            if attempt == retries:
                log.error(f"重试 {retries} 次仍失败：{type(e).__name__} {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(f"第 {attempt} 次失败：{type(e).__name__}，{delay}s 后重试")
            time.sleep(delay)
        except (AuthenticationError, BadRequestError) as e:
            log.error(f"不可恢复错误：{type(e).__name__} {e}")
            raise


def add_usage(usage):
    if not usage:
        return
    state["requests"] += 1
    state["prompt_tokens"] += usage.prompt_tokens or 0
    state["completion_tokens"] += usage.completion_tokens or 0
    log.info(
        f"tokens: prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} "
        f"total={usage.total_tokens}"
    )


def _do_request(kwargs):
    return client.chat.completions.create(**kwargs)


def chat(messages, temperature=0.7, max_tokens=None, stream=True):
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "timeout": 60,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if not stream:
        resp = with_retry(lambda: _do_request(kwargs))
        content = resp.choices[0].message.content
        print(content)
        add_usage(resp.usage)
        return content

    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}

    try:
        stream_resp = with_retry(lambda: _do_request(kwargs))
    except Exception:
        kwargs.pop("stream_options", None)
        stream_resp = with_retry(lambda: _do_request(kwargs))

    full = ""
    usage = None
    try:
        for chunk in stream_resp:
            if getattr(chunk, "usage", None):
                usage = chunk.usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full += delta
    except Exception as e:
        log.error(f"流式过程中断：{e}")

    print()
    add_usage(usage)
    return full


def print_help():
    print("""
命令：
  /exit            退出
  /clear           清空历史
  /history         查看历史
  /role <文本>     切换 system 角色
  /temp <数值>     设置 temperature（0~2）
  /max <整数>      设置 max_tokens，0 表示取消限制
  /stream on|off   开关流式输出
  /state           查看当前参数
  /cost            查看 token 统计
  /help            显示帮助
""")


def main():
    messages = [{"role": "system", "content": build_system_prompt()}]

    print("=" * 50)
    print(f"模型：{MODEL}")
    print("输入 /help 查看命令")
    print("=" * 50)

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("再见")
            break

        if user_input == "/clear":
            messages = [{"role": "system", "content": build_system_prompt()}]
            print("历史已清空\n")
            continue

        if user_input == "/history":
            if len(messages) <= 1:
                print("（暂无历史）\n")
            else:
                for m in messages:
                    content = m["content"]
                    if len(content) > 200:
                        content = content[:200] + "...(截断)"
                    print(f"[{m['role']}] {content}")
                print()
            continue

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/state":
            print(f"temperature = {state['temperature']}")
            print(f"max_tokens  = {state['max_tokens']}")
            print(f"stream      = {state['stream']}\n")
            continue

        if user_input == "/cost":
            total = state["prompt_tokens"] + state["completion_tokens"]
            print(f"请求次数   ：{state['requests']}")
            print(f"输入 tokens：{state['prompt_tokens']}")
            print(f"输出 tokens：{state['completion_tokens']}")
            print(f"总计       ：{total}\n")
            continue

        if user_input.startswith("/role "):
            new_role = user_input[6:].strip()
            if not new_role:
                print("用法：/role 你是海盗\n")
                continue
            messages[0] = {"role": "system", "content": new_role}
            print(f"角色已切换为：{new_role}\n")
            continue

        if user_input.startswith("/temp "):
            try:
                val = float(user_input[6:].strip())
                if not (0 <= val <= 2):
                    print("temperature 应在 0~2 之间\n")
                    continue
                state["temperature"] = val
                print(f"temperature = {val}\n")
            except ValueError:
                print("用法：/temp 0.7\n")
            continue

        if user_input.startswith("/max "):
            try:
                val = int(user_input[5:].strip())
                if val < 0:
                    print("max_tokens 不能为负\n")
                    continue
                state["max_tokens"] = None if val == 0 else val
                print(f"max_tokens = {state['max_tokens']}\n")
            except ValueError:
                print("用法：/max 100，或 /max 0 取消限制\n")
            continue

        if user_input.startswith("/stream "):
            arg = user_input[8:].strip().lower()
            if arg in ("on", "1", "true"):
                state["stream"] = True
                print("流式输出：开\n")
            elif arg in ("off", "0", "false"):
                state["stream"] = False
                print("流式输出：关\n")
            else:
                print("用法：/stream on 或 /stream off\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            print("AI：", end="", flush=True)
            answer = chat(
                messages,
                temperature=state["temperature"],
                max_tokens=state["max_tokens"],
                stream=state["stream"],
            )
            print()
        except Exception as e:
            print(f"\n出错：{e}\n")
            log.exception("对话失败")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
