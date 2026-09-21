import math
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    BM25Okapi = None


class _FallbackBM25:
    """Small dependency-free BM25 implementation used only when rank-bm25 is absent."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(map(len, corpus)) / self.n if self.n else 0.0
        self.k1 = 1.5
        self.b = 0.75
        df = {}
        for tokens in corpus:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        self.idf = {
            token: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def get_scores(self, query_tokens):
        scores = []
        for tokens in self.corpus:
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            dl = len(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                freq = tf[token]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else freq + self.k1
                score += self.idf.get(token, 0.0) * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class BM25Index:
    def __init__(self):
        self.rows = []
        self.bm25 = None

    def build(self, rows):
        self.rows = list(rows)
        tokenized = [re.findall(r"\b\w+\b", str(r[1]).lower()) for r in self.rows]
        if not tokenized:
            self.bm25 = None
        elif BM25Okapi is not None:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = _FallbackBM25(tokenized)

    def search(self, query, k=12, document_ids=None):
        if not self.bm25 or k <= 0:
            return []
        allowed = set(document_ids) if document_ids else None
        tokens = re.findall(r"\b\w+\b", query.lower())
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        out = []
        for i in order:
            row = self.rows[i]
            if allowed is not None and row[2] not in allowed:
                continue
            out.append(tuple(row))
            if len(out) >= k:
                break
        return out
