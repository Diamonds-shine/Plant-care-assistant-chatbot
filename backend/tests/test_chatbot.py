import pytest


@pytest.mark.parametrize(
    "question, expected_id",
    [
        # The six quick questions shown in the UI
        ("Why are my plant's leaves turning yellow?", "leaf_yellow"),
        ("How often should I water a money plant?", "kg_golden_pothos_water"),
        ("What plants grow well indoors?", "rec_beginner_plants"),
        ("How can I get rid of common plant pests?", "pest_general_treatment"),
        ("Why is my plant not growing?", "maint_not_growing"),
        ("How much sunlight does a snake plant need?", "kg_snake_plant_light"),
        # Paraphrases from the project brief
        ("My leaves are becoming yellow.", "leaf_yellow"),
        ("How much water does a pothos need?", "kg_golden_pothos_water"),
        ("There are tiny flies around the soil.", "pest_fungus_gnats"),
        ("The roots are coming out of the pot.", "maint_repot_when"),
        ("I think I gave my plant too much fertilizer.", "fert_too_much"),
        # Dataset-backed answers
        ("Is my ZZ plant safe for cats?", "kg_zz_plant_pets"),
        ("which plants are safe for my dog", "kg_list_pet_safe"),
    ],
)
def test_relevant_questions_find_expected_entry(bot, question, expected_id):
    result = bot.answer(question)
    assert not result.fallback
    assert result.matched_id == expected_id


def test_dark_room_question_gets_light_related_answer(bot):
    result = bot.answer("My snake plant sits in a dark room.")
    assert not result.fallback
    assert result.matched_id in {"kg_snake_plant_light", "kg_list_low_light"}


@pytest.mark.parametrize(
    "question",
    [
        "Who won the football match?",
        "Explain quantum computing.",
        "What is the capital of France?",
        "Write me a poem.",
        "What is today's stock market price?",
        "What is the stock price of a company named Green Plant?",
        "How does a power plant make electricity?",
        "asdfgh qwerty",
    ],
)
def test_unrelated_questions_get_fallback(bot, question):
    result = bot.answer(question)
    assert result.fallback
    assert result.topic is None


def test_no_meaningful_terms(bot):
    result = bot.answer("hello there")
    assert result.fallback and result.fallback_reason == "no_meaningful_terms"


def test_result_has_score_and_topic(bot):
    result = bot.answer("How do I get rid of aphids?")
    assert 0 < result.similarity <= 1
    assert result.topic == "pests_maintenance"
    assert result.topic_label == "Pests & Maintenance"
    assert result.confidence in {"high", "medium"}


def test_answers_are_deterministic(bot):
    first = bot.answer("brown spots on leaves")
    second = bot.answer("brown spots on leaves")
    assert (first.matched_id, first.similarity) == (second.matched_id, second.similarity)


def test_threshold_controls_fallback(bot):
    question = "leaves going yellow on my houseplant"
    assert not bot.answer(question, low_threshold=0.05).fallback
    assert bot.answer(question, low_threshold=1.01).fallback


def test_top_k_is_sorted(bot):
    from nlp.preprocess import preprocess

    scores = [m.score for m in bot.top_k(preprocess("yellow leaves"), k=5)]
    assert scores == sorted(scores, reverse=True)
