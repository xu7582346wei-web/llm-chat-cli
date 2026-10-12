"""FastAPI Web UI for LLM Agent"""
import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Optional
from fastapi import Depends, Header

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent as agent_module
from agent import (
    Agent, VectorStore, load_documents, list_sessions, SESSION_DIR,
)
from config import KNOWLEDGE_DIR, MODEL, EMBED_MODEL


# ==================== 全局状态 ====================
state = {
    "agent": None,
    "store": None,
    "lock": None,
}


# ==================== 启动 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["lock"] = asyncio.Lock()
    state["agent"] = Agent()

    store = VectorStore()
    chunks = load_documents(KNOWLEDGE_DIR)
    if chunks:
        print(f"[启动] 加载知识库 {len(chunks)} 个片段...")
        await asyncio.to_thread(store.add, chunks)
        print(f"[启动] 完成")
    else:
        print("[启动] 未找到知识库文档")

    agent_module.STORE = store
    state["store"] = store
    print(f"[启动] 模型：{MODEL}")
    yield
    print("[关闭] 服务停止")


app = FastAPI(title="LLM Agent API", lifespan=lifespan)

# 服务启动时间（用于健康检查）
START_TIME = time.time()

# API 认证（可选，通过 .env 配置 AGENT_API_KEY 启用）
API_KEY = os.getenv("AGENT_API_KEY", "")


async def verify_key(x_api_key: str = Header(None)):
    """API Key 认证。未配置 AGENT_API_KEY 时跳过（本地开发用）。"""
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")



# ==================== 请求模型 ====================
class ChatRequest(BaseModel):
    message: str


class SaveRequest(BaseModel):
    name: Optional[str] = None


class LoadRequest(BaseModel):
    name: str


# ==================== 状态 ====================
@app.get("/api/state")
async def get_state():
    store = state["store"]
    agent = state["agent"]
    return {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "knowledge_chunks": len(store.chunks) if store else 0,
        "history_len": len(agent.messages) if agent else 0,
    }


# ==================== 对话（SSE） ====================
@app.post("/api/chat", dependencies=[Depends(verify_key)])
async def chat(req: ChatRequest):
    if state["agent"] is None:
        raise HTTPException(500, "Agent 未初始化")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(kind, data):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    async def run_chat():
        try:
            async with state["lock"]:
                answer = await asyncio.to_thread(
                    state["agent"].chat,
                    req.message,
                    False,       # verbose
                    on_event,
                )
            await queue.put(("done", {"answer": answer}))
        except Exception as e:
            await queue.put(("error", {"message": str(e)}))
        finally:
            await queue.put(None)

    asyncio.create_task(run_chat())

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {kind}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==================== 清空 ====================
@app.post("/api/clear")
async def clear():
    state["agent"].clear()
    return {"ok": True}


# ==================== 历史 ====================
@app.get("/api/history")
async def history():
    msgs = state["agent"].messages
    return {
        "messages": [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in msgs
            if isinstance(m, dict) and m.get("content")
        ]
    }


# ==================== 会话 ====================
@app.get("/api/sessions")
async def sessions():
    return {
        "sessions": [
            {"name": p.name, "size": p.stat().st_size}
            for p in list_sessions()
        ]
    }


@app.post("/api/save")
async def save(req: SaveRequest):
    if req.name:
        path = SESSION_DIR / req.name
        if not path.suffix:
            path = path.with_suffix(".json")
        saved = state["agent"].save(path)
    else:
        saved = state["agent"].save()
    return {"path": str(saved)}


@app.post("/api/load")
async def load(req: LoadRequest):
    path = SESSION_DIR / req.name
    if not path.exists():
        path = SESSION_DIR / f"{req.name}.json"
    if not path.exists():
        raise HTTPException(404, "会话不存在")
    state["agent"].load(path)
    return {"ok": True, "messages": len(state["agent"].messages)}


# ==================== 静态文件 ====================
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")




@app.get("/health")
async def health():
    """健康检查接口。"""
    store = state.get("store")
    agent_inst = state.get("agent")
    return {
        "status": "ok",
        "model": MODEL,
        "knowledge_chunks": len(store.chunks) if store else 0,
        "history_len": len(agent_inst.messages) if agent_inst else 0,
        "uptime": round(time.time() - START_TIME, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
