import json
import re
import time
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.vector_store import AviationVectorStore

TOP_K = 5          # same number of chunks the pipeline sends to the llm
CANDIDATES = 20    # wider pool so hybrid and reranker have something to reorder
RRF_K = 60         # common default for reciprocal rank fusion
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    """Lowercase words, keeping AD numbers and part numbers like 2026-0161 as one token."""
    return re.findall(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", text.lower())


class Retrievers:
    """Three retrieval setups over the same indexed chunks, so only the method changes."""

    def __init__(self):
        self.vs = AviationVectorStore()
        self.vs.load_existing()

        store = self.vs.vector_store.get(include=["documents", "metadatas"])
        self.texts = store["documents"]
        self.metas = store["metadatas"]
        self.index_of = {self.chunk_key(m): i for i, m in enumerate(self.metas)}

        # ADs are full of exact identifiers that embeddings tend to blur, keyword search catches them
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.reranker = CrossEncoder(RERANKER_MODEL)

    @staticmethod
    def chunk_key(meta):
        return (meta["source_file"], meta["chunk_index"])

    def dense(self, question):
        results = self.vs.similarity_search_with_score(question, k=CANDIDATES)
        return [self.index_of[self.chunk_key(doc.metadata)] for doc, _ in results]

    def keyword(self, question):
        scores = self.bm25.get_scores(tokenize(question))
        return list(np.argsort(scores)[::-1][:CANDIDATES])

    def hybrid(self, question):
        # merge by rank position only, so bm25 and cosine scores never need to be on the same scale
        fused = {}
        for ranking in (self.dense(question), self.keyword(question)):
            for rank, idx in enumerate(ranking, 1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]

    def hybrid_rerank(self, question):
        # cross-encoder reads question and chunk together, slower but more precise than embeddings
        candidates = self.hybrid(question)
        scores = self.reranker.predict([(question, self.texts[i]) for i in candidates])
        return [candidates[i] for i in np.argsort(scores)[::-1]]

    def rank_of_expected(self, ranking, expected_file, expected_page):
        """Position (1 to TOP_K) of the first chunk from the expected page, or None if missed."""
        for rank, idx in enumerate(ranking[:TOP_K], 1):
            meta = self.metas[idx]
            if meta["source_file"] == expected_file and meta["page_number"] == expected_page:
                return rank
        return None


def summarize(ranks):
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hit5 = sum(1 for r in ranks if r is not None)
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return {"hit@1": hit1 / n, "hit@1_count": hit1, "hit@5": hit5 / n, "hit@5_count": hit5, "mrr": mrr, "n": n}


def run():
    with open("eval_dataset.json") as f:
        questions = [q for q in json.load(f) if q["in_scope"]]

    retrievers = Retrievers()
    setups = {
        "dense (baseline)": retrievers.dense,
        "hybrid": retrievers.hybrid,
        "hybrid + rerank": retrievers.hybrid_rerank,
    }

    results = {}
    for name, retrieve in setups.items():
        retrieve(questions[0]["question"])  # warm-up so model loading does not count as latency

        ranks, latencies = [], []
        for q in questions:
            start = time.perf_counter()
            ranking = retrieve(q["question"])
            latencies.append((time.perf_counter() - start) * 1000)
            ranks.append(retrievers.rank_of_expected(ranking, q["expected_file"], q["expected_page"]))

        results[name] = {**summarize(ranks), "median_ms": float(np.median(latencies)), "ranks": ranks}

    print(f"\n{len(questions)} in-scope questions, correct page must appear in top {TOP_K}\n")
    print(f"{'setup':<20}{'hit@1':>14}{'hit@5':>14}{'mrr':>8}{'median ms':>12}")
    for name, r in results.items():
        hit1 = f"{r['hit@1']:.2f} ({r['hit@1_count']}/{r['n']})"
        hit5 = f"{r['hit@5']:.2f} ({r['hit@5_count']}/{r['n']})"
        print(f"{name:<20}{hit1:>14}{hit5:>14}{r['mrr']:>8.2f}{r['median_ms']:>12.1f}")

    # per-question comparison is more honest than one average on a small test set
    baseline = results["dense (baseline)"]["ranks"]
    error_analysis = {}
    for name, r in results.items():
        if name == "dense (baseline)":
            continue
        fixed = [q["question"] for q, b, x in zip(questions, baseline, r["ranks"]) if b is None and x is not None]
        broke = [q["question"] for q, b, x in zip(questions, baseline, r["ranks"]) if b is not None and x is None]
        error_analysis[name] = {"fixed": fixed, "broke": broke}
        print(f"\n{name} vs baseline: fixed {len(fixed)}, broke {len(broke)}")
        for q in fixed:
            print(f"  fixed: {q}")
        for q in broke:
            print(f"  broke: {q}")

    still_missed = [q["question"] for q, x in zip(questions, results["hybrid + rerank"]["ranks"]) if x is None]
    print(f"\nstill missed by best setup: {len(still_missed)}")
    for q in still_missed:
        print(f"  {q}")

    report = {
        "questions": len(questions),
        "top_k": TOP_K,
        "setups": {name: {k: v for k, v in r.items()} for name, r in results.items()},
        "error_analysis": error_analysis,
        "still_missed": still_missed,
    }
    with open("retrieval_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved retrieval_results.json")


if __name__ == "__main__":
    run()
