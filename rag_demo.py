import os
import sys
import re
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

# 相似度阈值：低于这个值，认为不相关，不传给模型
SCORE_THRESHOLD = 0.5


# ============ Embedding ============
def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ============ 文档切分（改进版） ============
def chunk_text(text, chunk_size=500):
    """
    按 Markdown 结构切分：
    - 遇到 ## 开始新 chunk
    - 一级标题 # 作为语境前缀，附在每个 chunk 上
    - 跳过纯标题 chunk
    """
    lines = text.split("\n")
    chunks = []
    current_h1 = ""
    current_chunk = []

    def flush():
        if not current_chunk:
            return
        content = "\n".join(current_chunk).strip()
        if not content:
            return
        # 加一级标题语境
        if current_h1:
            full = f"{current_h1}\n{content}"
        else:
            full = content
        # 太长就切
        if len(full) <= chunk_size:
            chunks.append(full)
        else:
            start = 0
            while start < len(full):
                chunks.append(full[start:start + chunk_size])
                start += chunk_size - 50

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            # 一级标题
            flush()
            current_chunk = []
            current_h1 = stripped
        elif stripped.startswith("## "):
            # 二级标题，开始新 chunk
            flush()
            current_chunk = [line]
        else:
            current_chunk.append(line)
    flush()

    return [c for c in chunks if c.strip()]


def load_documents(folder="knowledge"):
    """加载所有 .md，切分，每个 chunk 带上文件来源。"""
    chunks = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for i, c in enumerate(chunk_text(text)):
            chunks.append({
                "text": f"[文件: {path.name}]\n{c}",
                "source": path.name,
                "chunk_id": i,
            })
    return chunks


# ============ 向量库 ============
class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None

    def add(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"[索引] 正在为 {len(texts)} 个片段生成向量...")
        vectors = embed(texts)
        self.vectors = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-10)
        print(f"[索引] 完成，向量维度：{self.vectors.shape}")

    def search(self, query, top_k=5, threshold=SCORE_THRESHOLD):
        if self.vectors is None or len(self.chunks) == 0:
            return []
        q_vec = np.array(embed([query])[0], dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        scores = self.vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            score = float(scores[i])
            if score < threshold:
                break  # 后面的更低，直接停
            results.append({**self.chunks[i], "score": score})
        return results


# ============ RAG 问答 ============
def rag_answer(query, store, top_k=5):
    hits = store.search(query, top_k=top_k)
    if not hits:
        return f"没有找到相关文档（相似度都低于 {SCORE_THRESHOLD}）。"

    print(f"\n[检索] 找到 {len(hits)} 个相关片段：")
    for h in hits:
        preview = h["text"].replace("\n", " ")[:70]
        print(f"  - [{h['source']}] 相似度 {h['score']:.3f}: {preview}...")

    context = "\n\n---\n\n".join(
        f"[来源：{h['source']}]\n{h['text']}" for h in hits
    )

    prompt = f"""请基于以下参考文档回答用户问题。
要求：
- 只使用参考文档里的信息
- 如果文档里没有，明确说"文档中没有相关信息"
- 不要编造

参考文档：
{context}

用户问题：{query}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============ 主程序 ============
if __name__ == "__main__":
    store = VectorStore()
    chunks = load_documents("knowledge")
    print(f"[加载] 共 {len(chunks)} 个片段\n")
    for c in chunks:
        print(f"  · [{c['source']}#{c['chunk_id']}] {c['text'][:60].replace(chr(10), ' ')}...")
    print()
    store.add(chunks)

    print(f"\n知识库已就绪（相似度阈值 {SCORE_THRESHOLD}）。输入 /exit 退出\n")
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        print(f"\nAI：{rag_answer(q, store)}\n")
