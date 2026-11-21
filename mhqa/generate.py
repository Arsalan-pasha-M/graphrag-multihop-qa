from .llm import chat_json

SYSTEM = """Answer the question using ONLY the provided context. The question may require
combining facts from several documents (multi-hop).
Return JSON: {"answer": "<short answer: entity, date, number or yes/no>",
"reasoning_path": ["step 1 ...", "step 2 ..."],
"citations": ["<exact document title>", ...]}
If the context is insufficient, answer "unknown"."""

def answer(question: str, passages: list, facts: list | None = None) -> dict:
    ctx = ""
    if facts:
        ctx += "KNOWLEDGE GRAPH FACTS:\n" + "\n".join(
            f"({f['head']}) -[{f['rel']}]-> ({f['tail']})  [doc: {f['doc']}]" for f in facts) + "\n\n"
    ctx += "PASSAGES:\n" + "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)
    words = len(ctx.split())
    try:
        res = chat_json(SYSTEM, f"{ctx}\n\nQUESTION: {question}")
    except Exception:
        res = None
    if isinstance(res, list) and res and isinstance(res[0], dict):
        res = res[0]
    if not isinstance(res, dict) or not isinstance(res.get("answer"), (str, int, float)):
        return {"answer": "", "reasoning_path": [], "citations": [], "valid": False, "context_words": words}
    steps, cites = res.get("reasoning_path", []), res.get("citations", [])
    return {"answer": str(res["answer"]),
            "reasoning_path": steps if isinstance(steps, list) else [str(steps)],
            "citations": cites if isinstance(cites, list) else [str(cites)],
            "valid": True, "context_words": words}
