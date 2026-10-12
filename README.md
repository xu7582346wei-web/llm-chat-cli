# LLM Agent CLI

> 一个从零手写的 LLM Agent，覆盖 工具调用 · 记忆 · 规划 · RAG · 多智能体 · MCP 全栈能力。

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1c3d5a?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 功能

| 能力 | 说明 |
|---|---|
| 工具调用 | Function Calling + Pydantic 参数校验 + 错误自愈 |
| 长期记忆 | JSON 持久化，跨会话记住用户偏好 |
| 任务规划 | Plan-and-Execute，复杂任务自动拆解子任务 |
| RAG | Markdown 切分 + 向量检索 + 相似度阈值 |
| 多轮对话 | 短期记忆 + 历史压缩 |
| ReAct 循环 | 显式思考 + 行动 + 观察 |
| 多智能体 | Supervisor 调度多专业 Agent |
| MCP 协议 | 工具标准化，可插拔 |
| 双 Web UI | Streamlit + FastAPI（响应式，支持手机） |
| 评估 | 9 个测试用例，成功率 100% |
| 安全 | 提示注入检测 + 脱敏 + 限流 |
| 一键部署 | Docker + Nginx + systemd |

---

## 架构

    用户
     |
     +-- CLI
     +-- Streamlit
     +-- FastAPI
          |
          v
    +---------------+
    |  Agent 核心    |
    |  ---------    |
    |  ReAct 循环    |
    |  Plan-Execute |
    |  Memory       |
    |  RAG          |
    +-------+-------+
            v
    +-------+-------+-------+
    v       v       v
   LLM     工具    向量库
  Ollama  calc..  knowledge/

---

## 快速开始

### 1. 环境准备

    git clone https://github.com/xu7582346wei-web/llm-chat-cli.git
    cd llm-chat-cli
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 2. 拉取模型

    ollama pull qwen3:8b
    ollama pull nomic-embed-text

### 3. 配置

    cp .env.example .env

默认指向本地 Ollama：

    OPENAI_API_KEY=ollama
    OPENAI_BASE_URL=http://localhost:11434/v1

也支持 DeepSeek / 通义千问 / OpenAI / vLLM 等 OpenAI 兼容接口。

---

## 使用

### 命令行

    python agent.py
    python agent.py -q "北京天气怎么样"
    python agent.py --model qwen3:14b
    python agent.py --no-rag
    python agent.py --save-session

交互命令：

- /help      显示帮助
- /exit      退出
- /clear     清空历史
- /history   查看历史
- /save      保存会话
- /load      加载会话
- /sessions  列出所有会话

### Web UI

Streamlit：

    ./run_web.sh
    # http://localhost:8501

FastAPI：

    ./run_api.sh
    # http://localhost:8000

### MCP 协议

    python mcp_server.py
    npx @modelcontextprotocol/inspector python mcp_server.py
    python agent_mcp.py

### LangGraph 版本

    python langgraph_demo.py    # 入门
    python lg_checkpoint.py     # 记忆
    python lg_approval.py       # 人工审批
    python lg_subgraph.py       # 子图
    python lg_multiagent.py     # 多智能体

### 评估

    python eval/run_eval.py

输出：

    总计：9/9 通过（100.0%）
    平均耗时：6.75s

---

## 项目结构

    llm-chat-cli/
    +-- agent.py                 # 主程序（工具+记忆+规划+RAG）
    +-- agent_mcp.py             # MCP 版 Agent
    +-- mcp_server.py            # MCP Server
    +-- config.py                # 配置
    +-- cli.py                   # 命令行参数
    +-- logger.py                # JSON 日志 + 脱敏
    +-- security.py              # 注入检测 + 限流
    +-- app_streamlit.py         # Streamlit UI
    +-- app_fastapi.py           # FastAPI 后端
    +-- static/index.html        # FastAPI 前端
    +-- langgraph_demo.py        # LangGraph 入门
    +-- lg_checkpoint.py         # 记忆
    +-- lg_approval.py           # 人工审批
    +-- lg_subgraph.py           # 子图
    +-- lg_multiagent.py         # 多智能体
    +-- rag_demo.py              # RAG 演示
    +-- knowledge/               # 知识库
    +-- eval/                    # 评估
    +-- Dockerfile
    +-- docker-compose.yml
    +-- deploy-agent.service     # systemd 模板
    +-- deploy-nginx.conf        # Nginx 模板

---

## 技术栈

- LLM：Ollama (qwen3:8b) / OpenAI 兼容接口
- Embedding：nomic-embed-text
- Agent 框架：手写 + LangGraph
- 协议：MCP (Model Context Protocol)
- Web：FastAPI + Streamlit
- 校验：Pydantic v2
- 日志：结构化 JSON + 脱敏
- 部署：Docker + Nginx + systemd

---

## 学习路线

    1 打基础（7 天）
       API 调用 · 多轮对话 · 流式 · 异常处理

    2 Agent 核心原理（14 天）
       工具调用 · ReAct · 记忆 · 规划 · RAG

    3 框架与生产化（14 天）
       LangGraph · 多智能体 · MCP · 评估 · 安全 · 部署

---

## 安全

- 提示注入检测（正则规则 + 日志记录）
- 敏感信息脱敏（API Key / 邮箱 / 手机号 / 身份证）
- 危险工具人工审批（LangGraph interrupt）
- 限流（滑动窗口）
- 日志 JSON 化，便于审计

---

## 部署

### Docker Compose

    docker compose up -d

### systemd

    sudo cp deploy-agent.service /etc/systemd/system/agent.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now agent

### Nginx 反向代理

    sudo cp deploy-nginx.conf /etc/nginx/sites-available/agent
    sudo ln -s /etc/nginx/sites-available/agent /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx

---

## 贡献

欢迎提 Issue 和 PR。如果这个项目对你有帮助，点个 Star 支持一下。

---

## License

MIT
