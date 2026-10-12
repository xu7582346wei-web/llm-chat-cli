"""命令行参数解析。"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Agent CLI - 工具 + 记忆 + 规划 + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python agent.py                     # 启动交互式对话
  python agent.py -q "北京天气"        # 单次提问
  python agent.py --model qwen3:14b   # 换模型
  python agent.py --no-rag            # 不加载知识库
  python agent.py --save-session      # 保存本次会话
        """,
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="单次提问（不进入交互模式）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖默认模型",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="不加载知识库",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="不使用长期记忆",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="退出时保存会话到 sessions/",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()
