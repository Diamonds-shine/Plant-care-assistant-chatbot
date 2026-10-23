"""Retrieval-based chatbot: TF-IDF vectors + cosine similarity.

How an answer is found
----------------------
1. At start-up every knowledge-base entry is turned into several small
   "documents": its main question, each alternative phrasing, and its
   keywords. All documents are preprocessed and a TfidfVectorizer is fitted
   on them ONCE.
2. For each user question we preprocess it the same way, turn it into a
   TF-IDF vector, and compute the cosine similarity with every document.
3. An entry's score is the best score of any of its documents.
4. The best entry is returned only if its score clears the confidence
   threshold; otherwise a polite fallback message is returned.

TF-IDF measures shared words (lexical overlap), not meaning. That is why the
knowledge base lists several phrasings per question, and why a fallback is
better than forcing the nearest answer for unrelated questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from knowledge_base import TOPIC_LABELS, load_knowledge_base
from nlp.preprocess import ProcessedText, preprocess, preprocess_text, stem_tokens

# Words that signal a clearly non-gardening request. A question that contains
# one of these AND none of the CARE_TERMS below is treated as out of domain,
# even if it mentions the word "plant" (e.g. "stock price of Green Plant Ltd").
OFF_DOMAIN_CUES = stem_tokens(
    """stock stocks share shares price prices market markets company companies
    invest investment bitcoin crypto football cricket match matches game games
    score scores tournament capital country countries president minister
    election poem poems song songs joke jokes story quantum physics computing
    computer programming python javascript code movie movies film films recipe
    recipes cook cooking bake restaurant weather forecast news celebrity
    homework essay translate salary job jobs power electricity energy factory
    manufacturing protein diet weight calories""".split()
)

# Stems that show a genuine plant-care intent.
CARE_TERMS = stem_tokens(
    """water watering overwatering underwatering soil pot repot repotting leaf
    leaves light sunlight sun shade fertilizer fertilize feed pest pests mite
    mites gnat gnats aphid aphids mealybugs rot root roots grow growing growth
    care yellow brown drooping wilting droop wilt curl curling spots humidity
    toxic pet pets prune propagate cutting bloom flower flowering dormant
    drainage succulent cactus""".split()
)

# General words that show the question is about plants at all. Plant names
# from the knowledge base are added to this set when the chatbot starts.
GENERAL_PLANT_TERMS = stem_tokens(
    """plant plants indoor garden gardening fern palm orchid ivy vine stem
    stems bud buds seed seeds houseplant greenery foliage""".split()
)


@dataclass
class Match:
    entry_id: str
    question: str
    topic: str
    score: float


@dataclass
class RetrievalResult:
    """Everything the API (or a test) needs to know about one answer."""

    answer: str
    topic: str | None
    topic_label: str | None
    matched_id: str | None
    matched_question: str | None
    similarity: float
    confidence: str  # "high" | "medium" | "none"
    fallback: bool
    fallback_reason: str | None
    source: str | None
    related_questions: list[str] = field(default_factory=list)
    processed: ProcessedText | None = None
    top_matches: list[Match] = field(default_factory=list)


class PlantCareChatbot:
    """Load the knowledge base, build the TF-IDF index and answer questions."""

    def __init__(
        self,
        kb_paths: tuple[Path, ...] = (config.CORE_KB_PATH, config.GENERATED_KB_PATH),
        low_threshold: float = config.LOW_CONFIDENCE_THRESHOLD,
        high_threshold: float = config.HIGH_CONFIDENCE_THRESHOLD,
    ):
        self.entries = load_knowledge_base(*kb_paths)
        self.entries_by_id = {entry["id"]: entry for entry in self.entries}
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Build the searchable documents and remember which entry each belongs to.
        documents: list[str] = []
        self.doc_to_entry: list[int] = []
        for entry_index, entry in enumerate(self.entries):
            texts = [entry["question"], *entry["alternatives"], " ".join(entry["keywords"])]
            for text in texts:
                clean = preprocess_text(text).clean
                if clean:
                    documents.append(clean)
                    self.doc_to_entry.append(entry_index)
        self.doc_to_entry_array = np.array(self.doc_to_entry)

        # Text is already lower-cased, tokenized and stemmed, so the vectorizer
        # simply splits on spaces. ngram_range=(1, 2) adds word pairs such as
        # "yellow leaf" and "snake plant"; sublinear_tf dampens repeated words.
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.document_count = len(documents)

        # Domain vocabulary: care words + general plant words + every plant
        # name/alias in the knowledge base ("monstera", "potho", "zz", ...).
        plant_name_stems: set[str] = set()
        for entry in self.entries:
            if entry.get("plant"):
                names = [entry["plant"], entry.get("scientific_name") or "", *entry["keywords"][:5]]
                plant_name_stems.update(preprocess_text(" ".join(names)).stems)
        self.domain_terms = set(CARE_TERMS) | set(GENERAL_PLANT_TERMS) | plant_name_stems

    # ------------------------------------------------------------------
    def score_entries(self, processed: ProcessedText) -> np.ndarray:
        """Cosine similarity of the question with every entry (best document per entry)."""
        query_vector = self.vectorizer.transform([processed.clean])
        doc_scores = cosine_similarity(query_vector, self.doc_matrix).ravel()
        entry_scores = np.zeros(len(self.entries))
        np.maximum.at(entry_scores, self.doc_to_entry_array, doc_scores)
        return entry_scores

    def top_k(self, processed: ProcessedText, k: int = 3) -> list[Match]:
        scores = self.score_entries(processed)
        # Stable sort on -score: equal scores keep knowledge-base order (core entries first).
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Match(self.entries[i]["id"], self.entries[i]["question"], self.entries[i]["topic"], float(scores[i]))
            for i in order
        ]

    def is_out_of_domain(self, processed: ProcessedText) -> bool:
        """Two simple rules, checked before trusting the similarity score:
        1. The question must contain at least one plant-related word.
        2. A clear non-gardening cue ("stock", "football", "electricity"...)
           wins unless a specific care word ("water", "soil"...) is present.
        """
        stems = set(processed.stems)
        if not stems & self.domain_terms:
            return True
        return bool(stems & set(OFF_DOMAIN_CUES)) and not (stems & set(CARE_TERMS))

    def related_questions(self, entry: dict, limit: int = 3) -> list[str]:
        related = [self.entries_by_id[i]["question"] for i in entry.get("related", []) if i in self.entries_by_id]
        return related[:limit]

    # ------------------------------------------------------------------
    def answer(self, question: object, low_threshold: float | None = None) -> RetrievalResult:
        """Answer one user question. Raises InvalidInputError for bad input."""
        threshold = self.low_threshold if low_threshold is None else low_threshold
        processed = preprocess(question)

        if not processed.stems:
            return self._fallback(processed, "no_meaningful_terms", config.NO_TERMS_MESSAGE, [])

        top = self.top_k(processed, k=3)
        if self.is_out_of_domain(processed):
            return self._fallback(processed, "out_of_domain", config.FALLBACK_MESSAGE, top)

        best = top[0]
        if best.score < threshold:
            return self._fallback(processed, "low_similarity", config.FALLBACK_MESSAGE, top)

        entry = self.entries_by_id[best.entry_id]
        return RetrievalResult(
            answer=entry["answer"],
            topic=entry["topic"],
            topic_label=TOPIC_LABELS[entry["topic"]],
            matched_id=entry["id"],
            matched_question=entry["question"],
            similarity=best.score,
            confidence="high" if best.score >= self.high_threshold else "medium",
            fallback=False,
            fallback_reason=None,
            source=entry["source"],
            related_questions=self.related_questions(entry),
            processed=processed,
            top_matches=top,
        )

    @staticmethod
    def _fallback(processed: ProcessedText, reason: str, message: str, top: list[Match]) -> RetrievalResult:
        return RetrievalResult(
            answer=message,
            topic=None,
            topic_label=None,
            matched_id=None,
            matched_question=None,
            similarity=top[0].score if top else 0.0,
            confidence="none",
            fallback=True,
            fallback_reason=reason,
            source=None,
            related_questions=[
                "Why are my plant's leaves turning yellow?",
                "How often should I water a snake plant?",
                "Which houseplants are safe for cats and dogs?",
            ],
            processed=processed,
            top_matches=top,
        )

    # ------------------------------------------------------------------
    def topics(self) -> list[dict]:
        """Topic names, labels, entry counts and two example questions each."""
        summary: dict[str, dict] = {}
        for entry in self.entries:
            item = summary.setdefault(
                entry["topic"],
                {"topic": entry["topic"], "label": TOPIC_LABELS[entry["topic"]], "count": 0, "examples": []},
            )
            item["count"] += 1
            if len(item["examples"]) < 2:
                item["examples"].append(entry["question"])
        return [summary[topic] for topic in TOPIC_LABELS if topic in summary]

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry["source"]] = counts.get(entry["source"], 0) + 1
        return counts
