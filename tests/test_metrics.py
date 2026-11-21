from mhqa.utils import exact_match, f1_score, norm_entity

def test_em_normalizes():
    assert exact_match("The Beatles.", "beatles") == 1.0

def test_f1_partial():
    assert 0 < f1_score("John F. Kennedy", "Kennedy") < 1

def test_entity_norm():
    assert norm_entity("  The  Beatles ") == "beatles"
