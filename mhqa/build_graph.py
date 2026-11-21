"""Rebuild Neo4j from cached triples with entity resolution. No LLM calls."""
from . import data
from .extract import _load
from .graph_store import GraphStore
from .resolve import resolve_cache

def main():
    items = list(data.corpus_from(data.load_sample()).items())
    cache = _load()
    resolved = resolve_cache(items, cache)
    gs = GraphStore(); gs.reset()
    for title, text in items:
        gs.add_doc(title, text, resolved.get(title, []))
    print(f"Triple coverage: {sum(t in cache for t, _ in items)}/{len(items)} paragraphs")
    print("Neo4j graph built:", gs.stats())
    gs.close()

if __name__ == "__main__":
    main()
