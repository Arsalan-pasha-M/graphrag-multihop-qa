import re, string, collections

def norm_entity(name: str) -> str:
    """Canonical entity id so 'The Beatles' and 'beatles' merge in the graph."""
    n = re.sub(r"\s+", " ", name.strip().lower())
    n = re.sub(r"^(the|a|an) ", "", n)
    return n.strip(string.punctuation + " ")

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))

def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def lenient_normalize(s: str) -> str:
    """Like normalize_answer, but punctuation/dashes become spaces instead of vanishing."""
    s = s.lower()
    s = "".join(" " if (ch in string.punctuation or ch in "\u2010\u2011\u2012\u2013\u2014\u2212") else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def lenient_em(pred: str, gold: str) -> float:
    return float(lenient_normalize(pred) == lenient_normalize(gold))

def lenient_f1(pred: str, gold: str) -> float:
    p, g = lenient_normalize(pred).split(), lenient_normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)
