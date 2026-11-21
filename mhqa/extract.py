import json, pathlib, threading
from .llm import chat_json, FatalLLMError

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "triples_cache.jsonl"
_lock = threading.Lock()
_cache = None

def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                try:
                    d = json.loads(line); _cache[d["title"]] = d["triples"]
                except Exception:
                    pass
    return _cache

def _save(title, triples):
    cache = _load()
    with _lock:
        cache[title] = triples
        CACHE.parent.mkdir(exist_ok=True)
        with CACHE.open("a") as f:
            f.write(json.dumps({"title": title, "triples": triples}) + "\n")

def _as_list(res, key):
    """Accept {"key": [...]}, a bare list, or any dict holding a list."""
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        if isinstance(res.get(key), list):
            return res[key]
        for v in res.values():
            if isinstance(v, list):
                return v
    return []

def _clean(raw):
    out = []
    for t in raw:
        if isinstance(t, dict) and all(isinstance(t.get(k), str) and t.get(k) for k in ("head", "relation", "tail")):
            out.append({"head": t["head"], "relation": t["relation"].lower(), "tail": t["tail"]})
    return out

def _k(s):
    return " ".join(str(s).lower().split())

BATCH_SYSTEM = """You extract knowledge-graph triples from several Wikipedia paragraphs.
Return JSON: {"results":[{"title":"<exact title as given>","triples":[{"head":"...","relation":"...","tail":"..."}]}]}
Rules:
- Exactly one result per paragraph, using the exact title given.
- Use full canonical entity names (resolve pronouns; the paragraph's title is its main subject).
- relation: short lowercase phrase, e.g. "directed by", "born in", "member of", "spouse of".
- Only facts stated in the paragraph. At most 12 triples per paragraph. Include dates and places as tails."""

def extract_batch(items: list) -> dict:
    """items: [(title, text), ...] -> {title: triples}. Paragraphs the model skips stay uncached."""
    cache = _load()
    todo = [(t, x) for t, x in items if t not in cache]
    if todo:
        prompt = "\n\n".join(f"### TITLE: {t}\n{x}" for t, x in todo)
        try:
            res = chat_json(BATCH_SYSTEM, prompt)
        except FatalLLMError:
            raise
        except Exception as e:
            print(f"[extract] batch failed: {e}")
            res = None
        got = {}
        for r in _as_list(res, "results"):
            if isinstance(r, dict) and isinstance(r.get("title"), str):
                got[_k(r["title"])] = _clean(_as_list(r, "triples"))
        for t, _ in todo:
            if _k(t) in got:
                _save(t, got[_k(t)])
    return {t: cache.get(t, []) for t, _ in items}

ENT_SYSTEM = """List the named entities (people, works, places, organizations, dates) mentioned in the
question, exactly as written. Return JSON: {"entities":["..."]}"""

def question_entities(question: str) -> list:
    res = chat_json(ENT_SYSTEM, question)
    return [e for e in _as_list(res, "entities") if isinstance(e, str)]
