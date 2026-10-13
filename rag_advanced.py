"""高级 RAG：BM25 + 向量 + RRF 融合 + LLM 重排序"""
import os
import sys
import re
import json
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


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = {}
        self.doc_lengths = []
        self.avg_dl = 0
        self.N = 0

    def _tokenize(self, text):
        try:
            import jieba
            return list(jieba.cut(text.lower()))
        except ImportError:
            return re.findall(r'\w+', text.lower())

    def fit(self, documents):
        self.corpus = documents
        self.N = len(documents)
        self.doc_freqs = {}
        self.doc_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self.avg_dl = sum(self.doc_lengths) / self.N if self.N else 1

    def score(self, query, doc_idx):
        tokens = self._tokenize(query)
        doc_tokens = self._tokenize(self.corpus[doc_idx])
        dl = self.doc_lengths[doc_idx]

        score = 0
        for token in tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf = doc_tokens.count(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += idf * numerator / denominator
        return score

    def search(self, query, top_k=5):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def chunk_text(text, chunk_size=400, overlap=50):
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                chunks.append(section[start:start + chunk_size])
                start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def load_documents(folder="knowledge"):
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


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank_with_llm(query, candidates, top_k=3):
    if len(candidates) <= top_k:
        return candidates

    prompt = f"对以下文档与问题的相关性打分（0-10）。\n\n问题：{query}\n\n文档：\n"
    for i, c in enumerate(candidates):
        prompt += f"\n[{i}] {c['text'][:200]}\n"
    prompt += '\n输出 JSON：{"scores": [分数1, 分数2, ...]}'

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        scores = data.get("scores", [])
        if len(scores) == len(candidates):
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:top_k]]
    except Exception as e:
        print(f"  [重排序失败] {e}")

    return candidates[:top_k]


class AdvancedRAG:
    def __init__(self):
        self.chunks = []
        self.vectors = None
        self.bm25 = None

    def add(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"[索引] 向量化 {len(texts)} 个片段...")
        vecs = embed(texts)
        self.vectors = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-10)

        print(f"[索引] 构建 BM25...")
        self.bm25 = BM25()
        self.bm25.fit(texts)
        print(f"[索引] 完成")

    def search(self, query, top_k=3):
        q_vec = np.array(embed([query])[0], dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        vec_scores = self.vectors @ q_vec
        vec_ranking = list(np.argsort(vec_scores)[::-1][:10])

        bm25_results = self.bm25.search(query, top_k=10)
        bm25_ranking = [i for i, _ in bm25_results]

        fused = reciprocal_rank_fusion([vec_ranking, bm25_ranking])
        top_ids = [doc_id for doc_id, _ in fused[:10]]

        candidates = [self.chunks[i] for i in top_ids]
        return rerank_with_llm(query, candidates, top_k=top_k)


def rag_answer(query, rag):
    hits = rag.search(query)
    if not hits:
        return "没有找到相关文档。"

    print(f"\n[检索] {len(hits)} 个片段：")
    for h in hits:
        print(f"  - [{h['source']}] {h['text'][:60].replace(chr(10), ' ')}...")

    context = "\n\n---\n\n".join(
        f"[来源：{h['source']}]\n{h['text']}" for h in hits
    )

    prompt = f"基于以下参考文档回答问题，不要编造。\n\n参考文档：\n{context}\n\n问题：{query}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    rag = AdvancedRAG()
    chunks = load_documents("knowledge")
    rag.add(chunks)

    print("\n高级 RAG（BM25 + 向量 + RRF + 重排序）已就绪\n")
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/exit", ""):
            break
        print(f"\nAI：{rag_answer(q, rag)}\n")
