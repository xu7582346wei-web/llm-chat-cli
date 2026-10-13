"""MCP Server：视觉工具，让文本 Agent 获得识图能力"""
import os
import base64
from pathlib import Path

from mcp.server import MCPServer
from openai import OpenAI

mcp = MCPServer(
    "Vision Tools",
    instructions="提供图像理解工具：描述图片、提取文字、分析图表。",
)

# 视觉模型客户端（默认用本地 Ollama 的 qwen2.5vl）
vision_client = OpenAI(
    api_key=os.getenv("VISION_API_KEY", os.getenv("OPENAI_API_KEY", "ollama")),
    base_url=os.getenv("VISION_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")),
)
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")


def _encode_image(path: str) -> str:
    """把图片编码为 base64 data URL。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    ext = p.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{ext};base64,{data}"


def _ask_vision(prompt: str, image_path: str) -> str:
    """通用视觉问答。"""
    try:
        data_url = _encode_image(image_path)
    except Exception as e:
        return f"图片读取失败：{e}"

    try:
        resp = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            timeout=120,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"视觉模型调用失败：{e}"


@mcp.tool()
def describe_image(image_path: str) -> str:
    """描述图片内容。用自然语言详细说明图片里有什么。

    Args:
        image_path: 图片文件的路径，如 '/home/xuwei/test.png'
    """
    return _ask_vision(
        "请详细描述这张图片的内容，包括主要对象、场景、颜色、文字等。用简洁的中文回答。",
        image_path,
    )


@mcp.tool()
def extract_text(image_path: str) -> str:
    """从图片中提取文字（OCR）。

    Args:
        image_path: 图片文件的路径
    """
    return _ask_vision(
        "请提取这张图片中的所有文字，保持原始格式和排版。只输出文字，不要多余的解释。",
        image_path,
    )


@mcp.tool()
def analyze_chart(image_path: str) -> str:
    """分析图表（折线图、柱状图、饼图等），提取数据和趋势。

    Args:
        image_path: 图片文件的路径
    """
    return _ask_vision(
        "请分析这张图表：说明图表类型、数据趋势、关键数值，并用简洁语言总结主要结论。",
        image_path,
    )


@mcp.tool()
def analyze_screenshot(image_path: str) -> str:
    """分析软件截图，识别报错、界面元素或功能。

    Args:
        image_path: 图片文件的路径
    """
    return _ask_vision(
        "这是一张软件界面的截图。请识别：1) 界面类型和功能；2) 是否有报错信息，如有请完整提取；3) 关键元素说明。",
        image_path,
    )


if __name__ == "__main__":
    mcp.run()
