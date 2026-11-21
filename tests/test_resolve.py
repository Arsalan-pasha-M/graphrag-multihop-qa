from mhqa.resolve import strip_paren, build_title_map, resolve_name, resolve_cache
from mhqa.utils import lenient_em, norm_entity

TITLES = ["Schofield Barracks", "8th Military Police Brigade (United States)"]

def test_strip_paren():
    assert strip_paren("8th Military Police Brigade (United States)") == "8th Military Police Brigade"

def test_location_suffix_resolves_to_title():
    m = build_title_map(TITLES)
    assert resolve_name("Schofield Barracks, Hawaii", m) == "Schofield Barracks"

def test_unqualified_name_resolves_to_disambiguated_title():
    m = build_title_map(TITLES)
    assert resolve_name("8th Military Police Brigade", m) == TITLES[1]

def test_unknown_name_unchanged():
    m = build_title_map(TITLES)
    assert resolve_name("Honolulu", m) == "Honolulu"

def test_resolve_cache_rewrites_both_ends():
    items = [(t, "") for t in TITLES]
    cache = {TITLES[1]: [{"head": TITLES[1], "relation": "based at", "tail": "Schofield Barracks, Hawaii"}]}
    out = resolve_cache(items, cache)
    assert out[TITLES[1]][0]["tail"] == "Schofield Barracks"
    assert out[TITLES[0]] == []

def test_lenient_em_dash_variants():
    assert lenient_em("Niger-Congo", "Niger\u2013Congo") == 1.0

def test_norm_entity_is_stable():
    assert norm_entity("The  Beatles") == norm_entity("beatles")
