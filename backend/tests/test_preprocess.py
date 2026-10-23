import pytest

from nlp import preprocess as pp
from nlp.preprocess import InvalidInputError, normalize, preprocess, preprocess_text


def test_normalize_matches_documented_example():
    assert normalize("WHY are my plant's leaves turning yellow??") == "why are my plants leaves turning yellow"


def test_case_and_whitespace_do_not_change_result():
    assert preprocess("  HOW often   should I WATER?  ").stems == preprocess("how often should i water").stems


def test_curly_apostrophes_are_normalized():
    assert normalize("plant’s leaves") == normalize("plant's leaves") == "plants leaves"


def test_punctuation_and_emoji_are_removed():
    assert normalize("Help!!! 🌱 my plant...") == "help my plant"


@pytest.mark.parametrize("text", ["I don't water it", "I dont water it", "I do not water it"])
def test_negation_is_kept(text):
    assert "not" in preprocess(text).filtered


def test_quantity_words_are_kept():
    assert {"too", "much"} <= set(preprocess("too much fertilizer").filtered)


def test_plural_and_singular_give_same_stem():
    assert preprocess("leaves roots plants").stems == preprocess("leaf root plant").stems


def test_synonyms_map_common_names():
    assert "potho" in preprocess("how do I water a money plant").stems
    assert "pilea" in preprocess("chinese money plant care").stems  # longer phrase wins


def test_synonym_replacement_is_not_applied_twice():
    assert preprocess("tiny flies in the soil").filtered.count("fungus") == 1


def test_unknown_words_pass_through():
    assert "xylophone" in preprocess("xylophone leaves").filtered


@pytest.mark.parametrize("bad", ["", "   ", None, 42, ["water"]])
def test_invalid_input_is_rejected(bad):
    with pytest.raises(InvalidInputError):
        preprocess(bad)


def test_too_long_input_is_rejected():
    with pytest.raises(InvalidInputError) as info:
        preprocess("water " * 200)
    assert info.value.code == "QUESTION_TOO_LONG"


def test_stopword_only_input_gives_no_terms():
    assert preprocess("the and of is").stems == []


def test_works_without_nltk_tokenizer_data(monkeypatch):
    """The regexp fallback tokenizer gives the same tokens as NLTK's word_tokenize."""
    from nltk.tokenize import RegexpTokenizer

    expected = preprocess_text("why are my snake plant leaves drooping").stems
    monkeypatch.setattr(pp, "_tokenize_impl", RegexpTokenizer(r"[a-z0-9]+").tokenize)
    assert preprocess_text("why are my snake plant leaves drooping").stems == expected


def test_bundled_stopwords_cover_nltk_list():
    assert {"the", "is", "and", "why"} <= pp._BUNDLED_STOPWORDS
    assert "not" not in pp.STOPWORDS
