"""Text preprocessing for GreenBuddy.

Pipeline (each step is a separate function so it can be shown in a demo):

    raw text
      -> validate_question()      reject non-strings, empty and over-long input
      -> normalize()              unicode, lower-case, contractions, punctuation
      -> apply_synonyms()         "money plant" -> "pothos", "bugs" -> "pests"
      -> tokenize()               NLTK tokenizer
      -> remove_stopwords()       NLTK stop-word list, but negations are KEPT
      -> stem_tokens()            NLTK Porter stemmer ("watering" -> "water")
      -> clean string used by the TF-IDF vectorizer

Nothing here downloads data at request time. NLTK resources are looked up
once at import. If `punkt_tab` or the `stopwords` corpus are missing, the
pipeline falls back to NLTK's RegexpTokenizer and a bundled copy of the same
English stop-word list, so the chatbot keeps working on a fresh machine.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

import config
from nlp.synonyms import PHRASE_SYNONYMS, WORD_SYNONYMS


class InvalidInputError(ValueError):
    """Raised when a question cannot be processed (wrong type, empty, too long)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# NLTK resources (checked once)
# ---------------------------------------------------------------------------

# Copy of NLTK's English stop-word list, used only if the corpus is not
# downloaded. Keeping an identical copy means results do not change.
_BUNDLED_STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you you're you've you'll you'd your
    yours yourself yourselves he him his himself she she's her hers herself it
    it's its itself they them their theirs themselves what which who whom this
    that that'll these those am is are was were be been being have has had
    having do does did doing a an the and but if or because as until while of
    at by for with about against between into through during before after above
    below to from up down in out on off over under again further then once here
    there when where why how all any both each few more most other some such no
    nor not only own same so than too very s t can will just don don't should
    should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
    doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn
    mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn
    wasn't weren weren't won won't wouldn wouldn't""".split()
)

# Words that NLTK lists as stop words but that change the meaning of a
# plant-care question ("NOT draining", "TOO much water", "MORE light").
KEEP_WORDS = frozenset(
    {"not", "no", "nor", "never", "without", "too", "more", "less", "most",
     "very", "few", "much", "enough", "against", "over", "under", "after", "before"}
)

# Extra filler words common in chat questions that carry no plant meaning.
EXTRA_STOPWORDS = frozenset(
    {"please", "pls", "plz", "hi", "hello", "hey", "thanks", "thank", "tell",
     "know", "want", "would", "could", "also", "really", "get", "got", "like",
     "anyone", "someone", "kindly", "ok", "okay", "u", "ur", "im", "ive", "id",
     "keep", "keeps", "kept", "keeping"}
)


def _load_stopwords() -> tuple[frozenset[str], str]:
    try:
        from nltk.corpus import stopwords

        words = frozenset(stopwords.words("english"))
        return words, "nltk_corpus"
    except LookupError:
        return _BUNDLED_STOPWORDS, "bundled_copy"


def _load_word_tokenizer():
    try:
        import nltk
        from nltk.tokenize import word_tokenize

        nltk.data.find("tokenizers/punkt_tab")
        return word_tokenize, "nltk_word_tokenize"
    except LookupError:
        regex_tokenizer = RegexpTokenizer(r"[a-z0-9]+")
        return regex_tokenizer.tokenize, "nltk_regexp_tokenizer"


_NLTK_STOPWORDS, _STOPWORD_SOURCE = _load_stopwords()
STOPWORDS = (_NLTK_STOPWORDS | EXTRA_STOPWORDS) - KEEP_WORDS
_tokenize_impl, _TOKENIZER_SOURCE = _load_word_tokenizer()
_stemmer = PorterStemmer()


def nltk_status() -> dict[str, str]:
    """Report which NLTK components are in use (shown in /health and docs)."""
    return {"tokenizer": _TOKENIZER_SOURCE, "stopwords": _STOPWORD_SOURCE, "stemmer": "nltk_porter"}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def validate_question(text: object, max_length: int = config.MAX_QUESTION_LENGTH) -> str:
    """Return the stripped question or raise InvalidInputError."""
    if not isinstance(text, str):
        raise InvalidInputError("INVALID_QUESTION", "The question must be text.")
    stripped = text.strip()
    if not stripped:
        raise InvalidInputError("INVALID_QUESTION", "Please enter a valid plant-care question.")
    if len(stripped) > max_length:
        raise InvalidInputError(
            "QUESTION_TOO_LONG",
            f"Please keep your question under {max_length} characters.",
        )
    return stripped


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})
_CONTRACTIONS = [
    (re.compile(r"\bcan'?t\b|\bcannot\b"), "can not"),
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\bain'?t\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    # Apostrophe-less forms typed in chat ("dont", "isnt", "doesnt").
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|should|could|would)nt\b"), r"\1 not"),
]
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, expand negative contractions and strip punctuation/emoji.

    "WHY are my plant's leaves turning yellow??" -> "why are my plants leaves turning yellow"
    """
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = text.replace("'", "")             # plant's -> plants
    text = text.replace("-", " ").replace("_", " ")
    text = _NON_ALPHANUMERIC.sub(" ", text)  # punctuation, emoji, symbols
    return _WHITESPACE.sub(" ", text).strip()


# One regex with the longest phrases first, applied in a single left-to-right
# pass: "chinese money plant" wins over "money plant", and replaced text is
# never rewritten a second time.
_PHRASE_REGEX = re.compile(
    r"\b(?:"
    + "|".join(re.escape(phrase) for phrase in sorted(PHRASE_SYNONYMS, key=len, reverse=True))
    + r")\b"
)


def apply_synonyms(normalized: str) -> str:
    """Map everyday names and misspellings onto the knowledge-base vocabulary."""
    text = _PHRASE_REGEX.sub(lambda match: PHRASE_SYNONYMS[match.group(0)], normalized)
    words = [WORD_SYNONYMS.get(word, word) for word in text.split()]
    return " ".join(words)


def tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens using NLTK."""
    return [token for token in _tokenize_impl(text) if token.isalnum()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop common function words but keep negations and quantity words."""
    return [token for token in tokens if token not in STOPWORDS]


# Irregular plurals the Porter stemmer does not unify with their singular.
_IRREGULAR = {"leaves": "leaf", "leafs": "leaf", "cacti": "cactus", "fungi": "fungus", "mice": "mouse"}


def stem_tokens(tokens: list[str]) -> list[str]:
    """Reduce words to a common stem: watering/watered -> water, plants -> plant."""
    return [_stemmer.stem(_IRREGULAR.get(token, token)) for token in tokens]


@dataclass
class ProcessedText:
    """Every intermediate stage, kept so the demo mode can show the pipeline."""

    original: str
    normalized: str
    with_synonyms: str
    tokens: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)

    @property
    def clean(self) -> str:
        """The space-joined stems given to the TF-IDF vectorizer."""
        return " ".join(self.stems)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "with_synonyms": self.with_synonyms,
            "tokens": self.tokens,
            "filtered": self.filtered,
            "stems": self.stems,
        }


def preprocess_text(text: str) -> ProcessedText:
    """Run the pipeline on already-validated text (also used for KB documents)."""
    normalized = normalize(text)
    with_synonyms = apply_synonyms(normalized)
    tokens = tokenize(with_synonyms)
    filtered = remove_stopwords(tokens)
    stems = stem_tokens(filtered)
    return ProcessedText(text, normalized, with_synonyms, tokens, filtered, stems)


def preprocess(question: object) -> ProcessedText:
    """Validate a user question and run the full pipeline."""
    return preprocess_text(validate_question(question))
