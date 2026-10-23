import config
from nlp.evaluator import load_dataset, run_evaluation


def test_dataset_has_both_splits_with_out_of_domain_questions():
    data = load_dataset()
    for split in ("calibration", "test"):
        assert len(data[split]) >= 30
        assert any(item["expected"] == "fallback" for item in data[split])


def test_evaluation_runs_and_config_matches_calibration(bot):
    report = run_evaluation(bot)
    assert report["chosen_threshold"] == config.LOW_CONFIDENCE_THRESHOLD
    metrics = report["test_result"]["metrics"]
    # Conservative floors so a regression is noticed, not targets to brag about.
    assert metrics["topic_accuracy_in_domain"] >= 0.75
    assert metrics["fallback_recall"] >= 0.8
    assert metrics["in_domain_precision"] >= 0.9
