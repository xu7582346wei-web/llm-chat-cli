"""
命令行参数解析模块。

使用 argparse 定义 agent.py 支持的所有命令行选项。
被 agent.py 的 main() 调用，返回解析后的 args 对象。
"""
import argparse


def parse_args():
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 包含以下字段：
            - query (str|None): 单次提问内容
            - model (str|None): 覆盖默认模型
            - no_rag (bool): 是否禁用知识库
            - no_memory (bool): 是否禁用长期记忆
            - save_session (bool): 退出时是否自动保存会话
            - log_level (str|None): 覆盖日志级别
    """
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

    # 单次提问模式：给 -q 就不进交互
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="单次提问（不进入交互模式）",
    )

    # 覆盖配置文件里的模型名
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖默认模型",
    )

    # 禁用 RAG（不加载 knowledge/ 目录）
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="不加载知识库",
    )

    # 禁用长期记忆（不注入 system prompt）
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="不使用长期记忆",
    )

    # 退出时自动保存会话
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="退出时保存会话到 sessions/",
    )

    # 覆盖日志级别
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    return parser.parse_args()
