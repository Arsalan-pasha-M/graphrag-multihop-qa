"""Build both indexes: Qdrant (vectors) and Neo4j (LLM-extracted knowledge graph)."""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from . import data
from .vector_store import VectorStore
from .graph_store import GraphStore
from .extract import extract_batch, _load
from .llm import FatalLLMError
from .resolve import resolve_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--type", default="bridge", choices=["bridge", "comparison", "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=6, help="paragraphs per LLM request")
    a = ap.parse_args()

    sample = data.build_sample(a.n, a.seed, a.type)
    corpus = data.corpus_from(sample)
    items = list(corpus.items())
    print(f"{len(sample)} questions -> {len(items)} unique paragraphs")

    vs = VectorStore(); vs.reset(); vs.index(corpus)
    print("Qdrant indexed.")

    cache = _load()
    try:
        for p in range(3):                       # up to 3 passes to fill gaps
            todo = [kv for kv in items if kv[0] not in cache]
            if not todo:
                break
            batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
            with ThreadPoolExecutor(a.workers) as ex:
                list(tqdm(ex.map(extract_batch, batches), total=len(batches),
                          desc=f"Extracting pass {p+1} ({len(todo)} paragraphs)"))
    except FatalLLMError as e:
        print(f"\nStopped: {e}\nProgress is saved in data/triples_cache.jsonl. "
              "Fix the problem (or wait), then re-run the same command to resume.")
        sys.exit(1)

    missing = [t for t, _ in items if t not in cache]
    print(f"Triple coverage: {len(items) - len(missing)}/{len(items)} paragraphs")
    if missing:
        print("Warning: some paragraphs got no triples; re-run the command to retry them.")

    gs = GraphStore(); gs.reset()
    resolved = resolve_cache(items, cache)
    for title, text in items:
        gs.add_doc(title, text, resolved.get(title, []))
    print("Neo4j graph built:", gs.stats())
    gs.close()

if __name__ == "__main__":
    main()
