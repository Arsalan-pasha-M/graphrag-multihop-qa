import time
from collections import Counter
from .vector_store import VectorStore
from .graph_store import GraphStore
from .extract import question_entities
from .generate import answer
from .utils import norm_entity
from . import config

class GraphRAG:
    """Hybrid retrieval: vector seeds + entity linking + multi-hop Neo4j traversal."""
    name = "graph_rag"
    def __init__(self, vs: VectorStore | None = None, gs: GraphStore | None = None):
        self.vs = vs or VectorStore()
        self.gs = gs or GraphStore()
        self.title_by_id = {norm_entity(t): t for t in self.gs.doc_titles()}

    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        seeds = self.vs.search(question, config.TOP_K)
        seed_titles = [s["title"] for s in seeds]

        # 1) link question entities to graph nodes (fallback: entities of top seed docs)
        ids = self.gs.find_entities(question_entities(question))
        if not ids:
            ids = self.gs.entities_of_docs(seed_titles[:2])

        # 2) multi-hop traversal (edges come back nearest-first)
        degs = self.gs.degrees(ids) if ids else {}
        facts, seen = [], set()
        for i in sorted(ids, key=lambda x: degs.get(x, 0)):      # most specific entity first
            for f in self.gs.neighborhood([i], hops=config.GRAPH_HOPS):
                k = (f["head"], f["rel"], f["tail"], f["doc"])
                if k not in seen:
                    seen.add(k); facts.append(f)
        facts = facts[: config.MAX_EDGES]

        # 3) bridge documents: an entity on a retrieved path that has its own document
        bridge = []
        for f in facts:
            for name in (f["head"], f["tail"]):
                t = self.title_by_id.get(norm_entity(name))
                if t and t not in seed_titles and t not in bridge:
                    bridge.append(t)
        edge_docs = [d for d, _ in Counter(f["doc"] for f in facts).most_common()]

        # 4) evidence = top vector hits, then bridge docs, then other graph docs
        ordered = list(dict.fromkeys(seed_titles[:3] + bridge + edge_docs + seed_titles[3:]))[: config.MAX_PASSAGES]
        texts = {s["title"]: s["text"] for s in seeds}
        texts.update(self.gs.doc_texts([t for t in ordered if t not in texts]))
        passages = [{"title": t, "text": texts[t]} for t in ordered if t in texts]

        out = answer(question, passages, facts)
        out.update(titles=[p["title"] for p in passages], facts=facts, entities=ids,
                   latency=time.perf_counter() - t0)
        return out
