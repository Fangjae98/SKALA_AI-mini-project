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
        if line.startswith(("- 기준", "- 적용 설비", ">")):   # 숫자 기준·설비 ID 목록·작성자 메모는 의미 검색을 흐림
            continue
        head.append(line.lstrip("#- ").strip())
    return " ".join(h for h in head if h)


# ── 상황별 절(## 상황 N: …) — 매뉴얼 하나에 여러 상황과 그 대응을 담는다 ─────────────
def split_runbook(text: str) -> tuple[str, list[dict]]:
    """매뉴얼을 (기본 부분, [상황 절들])로 나눈다. '## 상황 …'으로 시작하는 절이 상황이고,
    그 밖의 내용(제목·적용·기본 절차)은 기본 부분이다. 상황 절이 없는 옛 양식 매뉴얼은 통째로 기본 부분."""
    base, situations, cur = [], [], None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m and m.group(1).strip().startswith("상황"):
            title = re.sub(r"^상황\s*\d*\s*[:：.)]?\s*", "", m.group(1).strip())
            cur = {"title": title, "lines": []}
            situations.append(cur)
            continue
        if m:   # '## 기본 절차' 같은 다른 절은 기본 부분
            cur = None
        if line.startswith(">"):   # 작성자용 메모(> …)는 검색·보고서에 쓰지 않는다
            continue
        (cur["lines"] if cur else base).append(line)
    return ("\n".join(base).strip(),
            [{"title": s["title"], "body": "\n".join(s["lines"]).strip()} for s in situations])


def _uses_base_steps(body: str) -> bool:
    """상황 절의 '- 대응:' 줄에 '기본 절차'가 있으면 기본 절차를 함께 따른다(동일 대처 또는 기본 + 추가)."""
    line = next((l for l in body.splitlines() if l.lstrip("- ").startswith("대응")), "")
    if "대신" in line:   # "아래 절차(기본 절차 대신)"
        return False
    return "기본 절차" in line or not re.search(r"^\d+\.\s", body, flags=re.M)


def data_normal_situation(text: str) -> str | None:
    """'- 사용 조건: 데이터 정상'이 적힌 상황 — 운영자는 이상을 말하는데 계측값은 기준 이내일 때 쓰는 절차."""
    return next((s["title"] for s in split_runbook(text)[1] if "데이터 정상" in _field(s["body"], "사용 조건")), None)


def runbook_excerpt(text: str, situation: str | None = None) -> str:
    """보고서 작성에 넘길 매뉴얼 발췌 — 기본 부분 + (찾은 상황이 있으면) 그 상황 절만.
    다른 상황 절까지 다 넘기면 모델이 질문과 무관한 상황의 조치를 섞어 쓰기 때문이다."""
    base, situations = split_runbook(text)
    hit = next((s for s in situations if s["title"] == situation), None)
    if not situations or not hit:
        return base if situations else text
    if not _uses_base_steps(hit["body"]):   # 별도 절차만 따르는 상황 — 기본 절차 번호 목록은 뺀다
        base = "\n".join(l for l in base.splitlines()
                         if not re.match(r"^\d+\.\s", l) and not l.startswith("## 기본 절차")).strip()
        base = re.sub(r"\n{3,}", "\n\n", base)
    return f"{base}\n\n## 해당 상황: {hit['title']}\n{hit['body']}"


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
def _field(body: str, name: str) -> str:
    line = next((l for l in body.splitlines() if l.lstrip("- ").startswith(name)), "")
    return line.split(":", 1)[1].strip() if ":" in line else ""


class RunbookIndex:
    """매뉴얼 색인. 매뉴얼마다 '기본 부분' 1개 + '상황 절' 수만큼 벡터를 만든다(절 단위 검색).

    - 매뉴얼 점수 = 기본 부분의 점수(키워드가 질문에 직접 있는 상황 절은 그 점수도 반영). 최종 선택은 selection.py
    - 상황 고르기(choose_situation):
      ① 상황 절의 '- 키워드:' 단어가 질문에 있으면 그 상황(파일 위쪽 상황 먼저 — 화재처럼 위험한 상황을 위에 둘 것)
      ② 아니면 의미 점수가 기본 부분보다 rag.situation_margin 이상 높은 상황
      ③ 둘 다 아니면 None → 기본 절차만 사용(질문에 원인 단서가 없는데 특정 원인을 단정하지 않기 위해)"""

    def __init__(self, embedder=None, runbook_dir: Path = RUNBOOK_DIR):
        self.embedder = embedder or make_embedder()
        self.items = []
        for path in sorted(runbook_dir.glob("*.md")):
            text = path.read_text()
            title = text.splitlines()[0].lstrip("# ").strip()
            base, situations = split_runbook(text)
            self.items.append({"incident_type": path.stem, "title": title, "situation": None,
                               "keywords": [], "header": runbook_header(base)})
            for sit in situations:
                body = "\n".join(l for l in sit["body"].splitlines()
                                 if not l.lstrip("- ").startswith(("대응", "키워드")))
                keywords = [k.strip() for k in re.split(r"[,，]", _field(sit["body"], "키워드")) if k.strip()]
                self.items.append({"incident_type": path.stem, "title": title, "situation": sit["title"],
                                   "keywords": keywords,
                                   # 상황 절은 매뉴얼 제목 없이 상황 내용만 벡터로 — 제목이 같으면 절끼리 점수가 비슷해짐
                                   "header": f"{sit['title']} {runbook_header(body)}"})
        headers = [it["header"] for it in self.items]
        self.embedder.fit(headers)
        self.vectors = self.embedder.embed(headers)

    def _scores(self, query: str) -> list[tuple[dict, float]]:
        qv = self.embedder.embed([query])[0]
        return [(it, round(self.embedder.similarity(qv, vec), 4)) for it, vec in zip(self.items, self.vectors)]

    def choose_situation_by(self, query: str, incident_type: str, scored=None) -> tuple[str | None, str | None]:
        """(상황 이름, 고른 방법 'keyword'|'semantic'|None). 키워드 = 운영자가 직접 말한 관찰(연기·에어컨 꺼짐 등)."""
        scored = scored or self._scores(query)
        mine = [(it, sc) for it, sc in scored if it["incident_type"] == incident_type]
        q = query.lower()
        for it, _ in mine:
            if it["situation"] and any(k.lower() in q for k in it["keywords"]):
                return it["situation"], "keyword"
        base = next((sc for it, sc in mine if it["situation"] is None), 0.0)
        sits = [(it, sc) for it, sc in mine if it["situation"]]
        if not sits:
            return None, None
        it, sc = max(sits, key=lambda x: x[1])
        margin = RULES["rag"].get("situation_margin", {}).get(getattr(self.embedder, "name", "tfidf"), 0.05)
        return (it["situation"], "semantic") if sc - base >= margin else (None, None)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        scored = self._scores(query)
        best: dict[str, dict] = {}
        q = query.lower()
        for it, score in scored:
            # 매뉴얼 선택 점수는 기본 부분으로 매긴다. 상황 절은 키워드가 질문에 직접 있을 때만 반영
            # (절이 많을수록 '충전기'→'충전 중'처럼 우연히 겹치는 단어로 엉뚱한 매뉴얼이 뽑히기 때문)
            keyword_hit = bool(it["situation"]) and any(k.lower() in q for k in it["keywords"])
            if it["situation"] and not keyword_hit:
                continue
            if keyword_hit:   # 상황 키워드가 질문에 그대로 있음 = 이 매뉴얼을 가리키는 분명한 신호
                score = round(score + RULES["rag"].get("keyword_bonus", 0.0), 4)
            cur = best.get(it["incident_type"])
            if cur is None or score > cur["score"]:
                stage = RULES["incident_types"].get(it["incident_type"], {}).get("stage", "")
                best[it["incident_type"]] = {"incident_type": it["incident_type"], "title": it["title"],
                                             "stage": stage, "score": score}
        hits = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        for h in hits:
            h["situation"], h["situation_by"] = self.choose_situation_by(query, h["incident_type"], scored)
        return hits
