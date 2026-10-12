# LLM Chat CLI

一个基于大语言模型 API 的命令行聊天机器人。

## 功能

- 多轮对话，带上下文记忆
- 流式输出
- 可调 temperature、max_tokens
- 动态切换 system 角色
- 异常重试、超时、日志
- Token 用量统计
- 当前时间注入

## 安装

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 配置

复制 .env.example 为 .env：

    cp .env.example .env
    vim .env

支持任何 OpenAI 兼容接口。

## 运行

    python chat_multi.py

## 命令

- /help 显示帮助
- /exit 退出
- /clear 清空历史
- /history 查看历史
- /role 切换角色
- /temp 设置 temperature
- /max 设置 max_tokens
- /stream on|off 开关流式
- /state 查看参数
- /cost 查看 token 统计
