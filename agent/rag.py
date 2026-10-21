"""[After 추가 — RAG] 대응 매뉴얼 의미 검색.

Before는 알람 문장에서 키워드를 '목록 순서대로' 찾아 매뉴얼을 하나 골랐다. 그래서 알람이 여러 개면
질문과 상관없는 매뉴얼이 먼저 걸렸다(예: DR을 물었는데 피크 매뉴얼).
After는 매뉴얼 9종을 미리 벡터(숫자 목록)로 바꿔 두고, 사용자 질문도 벡터로 바꿔서 '의미가 가장 가까운'
매뉴얼을 찾는다. 이 파일은 MCP 서버(mcp_server.py)의 search_runbooks Tool이 사용한다.

임베딩(글 → 벡터) 방식은 3가지이고 config/rules.json의 rag.embedding_backend로 고른다(auto = LLM_BACKEND 따라감).
- tfidf  : 인터넷·모델 없이 동작하는 단어 기반 방식(fake 모드·테스트용). 결과가 항상 같다.
- ollama : 로컬 Ollama 임베딩 모델(기본 bge-m3, 한국어 지원). `ollama pull bge-m3` 필요.
- openai : OpenAI 임베딩 API(기본 text-embedding-3-small). 과금.

매뉴얼이 9개뿐이라 별도 벡터 DB 없이 메모리에 보관한다(매뉴얼이 수백~수천 개로 늘면 Chroma·FAISS 같은
벡터 DB로 바꾸는 것이 맞다 — 한계로 기록).
"""
import math
import os
import re
from collections import Counter
from pathlib import Path

import httpx

from .config import RULES, embedding_backend

RUNBOOK_DIR = Path(__file__).resolve().parent.parent / "data" / "runbooks"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


def runbook_header(text: str) -> str:
    """매뉴얼에서 검색에 쓸 부분(제목·단계·적용 상황·증상 예시)만 잘라낸다.
    번호 절차(1. 2. …)는 매뉴얼마다 비슷한 단어가 많아 검색을 흐리므로 제외한다."""
    head = []
    for line in text.splitlines():
        if re.match(r"^\d+\.\s", line):
            break
        if line.startswith("- 기준"):
            continue
        head.append(line.lstrip("#- ").strip())
    return " ".join(h for h in head if h)


# ── 임베딩 방식 3종 ───────────────────────────────────────────────────────
class TfidfEmbedder:
    """단어 + 글자 2개 묶음(바이그램)의 TF-IDF 벡터. 한국어는 조사가 붙어 단어가 달라지므로
    ('온도가', '온도를') 글자 묶음으로도 비교해 '온도'를 같은 것으로 잡는다."""
    name = "tfidf"

    @staticmethod
    def _features(text: str) -> Counter:
        words = re.findall(r"[가-힣a-z0-9]+", text.lower())
        feats = Counter(words)
        for w in words:
            feats.update(w[i:i + 2] for i in range(len(w) - 1))
        return feats

    def fit(self, docs: list[str]) -> None:
        df = Counter()
        for d in docs:
            df.update(set(self._features(d)))
        n = len(docs)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1 for f, c in df.items()}

    def embed(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            feats = self._features(t)
            vec = {f: c * self.idf[f] for f, c in feats.items() if f in self.idf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            out.append({f: v / norm for f, v in vec.items()})
        return out

    @staticmethod
    def similarity(a: dict, b: dict) -> float:
        return sum(v * b.get(f, 0.0) for f, v in a.items())


class _DenseEmbedder:
    def fit(self, docs: list[str]) -> None:   # 사전 학습된 모델이라 따로 맞출 것이 없다
        pass

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)


class OllamaEmbedder(_DenseEmbedder):
    name = "ollama"

    def __init__(self, model: str):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(OLLAMA_EMBED_URL, json={"model": self.model, "input": texts}, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["embeddings"]


class OpenAIEmbedder(_DenseEmbedder):
    name = "openai"

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(OPENAI_EMBED_URL, json={"model": self.model, "input": texts},
                          headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60.0)
        resp.raise_for_status()
        return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]


def make_embedder(backend: str | None = None):
    backend = backend or embedding_backend()
    if backend == "ollama":
        return OllamaEmbedder(RULES["rag"]["ollama_model"])
    if backend == "openai":
        return OpenAIEmbedder(RULES["rag"]["openai_model"])
    return TfidfEmbedder()


# ── 매뉴얼 색인 ──────────────────────────────────────────────────────────
class RunbookIndex:
    def __init__(self, embedder=None, runbook_dir: Path = RUNBOOK_DIR):
        self.embedder = embedder or make_embedder()
        self.items = []
        for path in sorted(runbook_dir.glob("*.md")):
            text = path.read_text()
            title = text.splitlines()[0].lstrip("# ").strip()
            self.items.append({"incident_type": path.stem, "title": title, "header": runbook_header(text)})
        headers = [it["header"] for it in self.items]
        self.embedder.fit(headers)
        self.vectors = self.embedder.embed(headers)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        qv = self.embedder.embed([query])[0]
        scored = []
        for it, vec in zip(self.items, self.vectors):
            stage = RULES["incident_types"].get(it["incident_type"], {}).get("stage", "")
            scored.append({"incident_type": it["incident_type"], "title": it["title"], "stage": stage,
                           "score": round(self.embedder.similarity(qv, vec), 4)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]
