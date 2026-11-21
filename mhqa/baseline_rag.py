import time
from .vector_store import VectorStore
from .generate import answer
from . import config

class BaselineRAG:
    name = "baseline_rag"
    def __init__(self, vs: VectorStore | None = None):
        self.vs = vs or VectorStore()

    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        passages = self.vs.search(question, config.TOP_K)
        out = answer(question, passages)
        out.update(titles=[p["title"] for p in passages], facts=[], latency=time.perf_counter() - t0)
        return out
