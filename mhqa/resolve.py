"""Entity resolution: map extracted entity names onto document titles."""
import re
from .utils import norm_entity

def strip_paren(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)?\s*$", "", s).strip()

def build_title_map(titles):
    m = {}
    for t in titles:
        m.setdefault(norm_entity(t), t)
    for t in titles:
        m.setdefault(norm_entity(strip_paren(t)), t)
    return m

def resolve_name(name: str, m: dict) -> str:
    for cand in (name, name.split(",")[0], strip_paren(name)):
        k = norm_entity(cand)
        if k in m:
            return m[k]
    return name

def resolve_cache(items, cache):
    m = build_title_map([t for t, _ in items])
    out = {}
    for title, _ in items:
        out[title] = [{"head": resolve_name(t["head"], m), "relation": t["relation"],
                       "tail": resolve_name(t["tail"], m)} for t in cache.get(title, [])]
    return out
