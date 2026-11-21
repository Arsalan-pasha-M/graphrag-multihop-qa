import json, random, pathlib
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "hotpot_dev_distractor_v1.json"
SAMPLE = DATA / "sample.json"

def build_sample(n: int, seed: int = 42, qtype: str = "bridge"):
    """Reproducible sample of multi-hop questions (bridge = needs chaining across docs)."""
    data = json.loads(RAW.read_text())
    if qtype != "all":
        data = [d for d in data if d["type"] == qtype]
    random.Random(seed).shuffle(data)
    sample = data[:n]
    SAMPLE.write_text(json.dumps(sample))
    return sample

def load_sample():
    return json.loads(SAMPLE.read_text())

def corpus_from(sample):
    """Shared corpus: title -> paragraph text (gold + distractors from all questions)."""
    corpus = {}
    for ex in sample:
        for title, sents in ex["context"]:
            corpus.setdefault(title, " ".join(s.strip() for s in sents))
    return corpus

def gold_titles(ex):
    return {t for t, _ in ex["supporting_facts"]}
