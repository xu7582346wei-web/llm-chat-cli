# LLM Agent CLI

基于本地大模型的命令行 Agent，支持工具调用、记忆、规划、RAG。

## 功能

- 多轮对话，带上下文记忆
- Function Calling：calculator、get_time、get_weather、search_knowledge
- 长期记忆：跨会话记住用户偏好
- 任务规划：复杂任务自动拆解
- RAG：基于本地知识库的文档问答
- 会话保存 / 加载
- 命令行参数

## 安装

1. pip install -r requirements.txt
2. ollama pull qwen3:8b
3. ollama pull nomic-embed-text
4. cp .env.example .env

## 运行

python agent.py

## 命令行参数

- -q "..."           单次提问
- --model qwen3:14b  换模型
- --no-rag           不加载知识库
- --no-memory        不用长期记忆
- --save-session     退出时保存会话

## 交互命令

- /help      显示帮助
- /exit      退出
- /clear     清空历史
- /history   查看历史
- /save      保存会话
- /load      加载会话
- /sessions  列出会话

## 项目结构

- agent.py       主程序
- config.py      配置
- cli.py         命令行参数
- logger.py      日志
- knowledge/     知识库
- sessions/      会话文件
