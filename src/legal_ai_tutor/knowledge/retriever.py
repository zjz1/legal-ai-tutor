# -*- coding: utf-8 -*-
"""混合检索：BM25（精确命中法条号/法律名） + 哈希向量（语义近似），用 RRF 融合。

不引入 jieba / 向量库依赖：中文按「单字 + 相邻二元组」切词，
在小规模法条库上召回已足够，后续可无痛替换为 BGE-M3 + Chroma。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import LawArticle, LawCorpus

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    tokens = [w.lower() for w in _LATIN.findall(text)]
    chars = _CJK.findall(text)
    tokens.extend(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:], strict=False))
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in documents]
        self.avg_len = (sum(self.doc_len) / len(documents)) if documents else 0.0
        self.term_freqs = [Counter(doc) for doc in documents]
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        results = [0.0] * len(self.documents)
        if not self.documents or not query:
            return results
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[index] / (self.avg_len or 1))
                results[index] += idf * freq * (self.k1 + 1) / (denom or 1)
        return results


class HashingEmbedder:
    """确定性哈希向量，零依赖、可离线、可复现。"""

    dim = 128

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[slot] += sign
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


@dataclass
class Retrieved:
    article: LawArticle
    score: float
    source: str


class HybridRetriever:
    RRF_K = 60

    def __init__(self, corpus: LawCorpus, embedder: HashingEmbedder | None = None) -> None:
        self.corpus = corpus
        self.embedder = embedder or HashingEmbedder()
        texts = [f"{item.law} 第{item.article}条 {item.text}" for item in corpus.articles]
        self._texts = texts
        self._bm25 = BM25([tokenize(t) for t in texts])
        self._vectors = self.embedder.embed(texts)

    @property
    def size(self) -> int:
        return len(self.corpus.articles)

    def search(self, query: str, top_k: int = 8) -> list[Retrieved]:
        if not self.corpus.articles:
            return []

        bm_scores = self._bm25.scores(tokenize(query))
        query_vector = self.embedder.embed([query])[0]
        vector_scores = [cosine(query_vector, vec) for vec in self._vectors]

        bm_rank = self._rank(bm_scores)
        vec_rank = self._rank(vector_scores)

        fused: list[Retrieved] = []
        for index, article in enumerate(self.corpus.articles):
            score = 0.0
            source = ""
            if index in bm_rank:
                score += 1.0 / (self.RRF_K + bm_rank[index])
                source = "bm25"
            if index in vec_rank:
                score += 1.0 / (self.RRF_K + vec_rank[index])
                source = "hybrid" if source else "vector"
            if score > 0:
                fused.append(Retrieved(article=article, score=score, source=source))

        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    @staticmethod
    def _rank(scores: list[float]) -> dict[int, int]:
        ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {index: rank + 1 for rank, index in enumerate(ordered) if scores[index] > 0}

    def as_context(self, retrieved: list[Retrieved], limit: int = 4) -> str:
        """把检索结果拼成给模型看的候选条文，带条号与生效日期。"""
        lines: list[str] = []
        for item in retrieved[:limit]:
            article = item.article
            suffix = f"（{article.effective_date} 施行）" if article.effective_date else ""
            lines.append(f"- {article.label}{suffix}：{article.text}")
        return "\n".join(lines)
