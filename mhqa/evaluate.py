"""Benchmark Baseline RAG vs GraphRAG: accuracy, retrieval quality, latency, reliability."""
import argparse, json, pathlib
import pandas as pd
from tqdm import tqdm
from . import data
from .utils import exact_match, f1_score
from .vector_store import VectorStore
from .graph_store import GraphStore
from .baseline_rag import BaselineRAG
from .graph_rag import GraphRAG

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

def score(ex, out):
    gold = data.gold_titles(ex)
    got = set(out["titles"])
    cites = set(out["citations"]) if isinstance(out["citations"], list) else set()
    return {
        "id": ex["_id"], "type": ex["type"], "level": ex["level"],
        "question": ex["question"], "gold": ex["answer"], "pred": out["answer"],
        "em": exact_match(out["answer"], ex["answer"]),
        "f1": f1_score(out["answer"], ex["answer"]),
        "support_recall": len(gold & got) / len(gold),   # fraction of gold docs retrieved
        "full_support": float(gold <= got),
        "full_support_evidence": float(gold <= (got | {f["doc"] for f in out["facts"]})),              # ALL gold docs retrieved (key multi-hop metric)
        "citation_precision": len(cites & gold) / len(cites) if cites else 0.0,
        "latency": out["latency"], "valid_json": float(out["valid"]),
        "context_words": out["context_words"], "n_facts": len(out["facts"]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N questions")
    a = ap.parse_args()
    sample = data.load_sample()[a.offset : a.offset + a.n]
    vs, gs = VectorStore(), GraphStore()
    systems = [BaselineRAG(vs), GraphRAG(vs, gs)]
    rows, raw = [], []
    for sys_ in systems:
        for ex in tqdm(sample, desc=sys_.name):
            try:
                out = sys_.run(ex["question"])
            except Exception:  # crashes count as failures
                out = {"answer": "", "citations": [], "titles": [], "facts": [], "latency": 0.0,
                       "valid": False, "context_words": 0, "reasoning_path": []}
            r = score(ex, out); r["system"] = sys_.name
            rows.append(r); raw.append({**r, "reasoning_path": out["reasoning_path"], "facts": out["facts"][:15]})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "per_question.csv", index=False)
    (RESULTS / "raw_outputs.json").write_text(json.dumps(raw, indent=1))

    agg = df.groupby("system").agg(
        EM=("em", "mean"), F1=("f1", "mean"), support_recall=("support_recall", "mean"),
        full_support=("full_support", "mean"),
        full_support_evidence=("full_support_evidence", "mean"), citation_precision=("citation_precision", "mean"),
        avg_latency_s=("latency", "mean"), p95_latency_s=("latency", lambda x: x.quantile(.95)),
        invalid_json_rate=("valid_json", lambda x: 1 - x.mean()), avg_context_words=("context_words", "mean"),
    ).round(3)
    by_level = df.groupby(["system", "level"])[["em", "f1", "full_support"]].mean().round(3)
    report = f"# Evaluation (n={len(sample)})\n\n{agg.T.to_markdown()}\n\n## By difficulty\n\n{by_level.to_markdown()}\n"
    (RESULTS / "report.md").write_text(report)
    print(report)
    gs.close()

if __name__ == "__main__":
    main()
