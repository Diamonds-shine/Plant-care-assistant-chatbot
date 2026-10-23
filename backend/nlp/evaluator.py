"""Evaluate GreenBuddy's retrieval and calibrate the confidence threshold.

Run from the backend folder:

    python3 -m nlp.evaluator

Method
------
* evaluation/eval_dataset.json has two labelled splits written independently
  of the knowledge base: "calibration" and "test".
* Each question is either in-domain (expected: "answer", with a list of
  acceptable entry ids) or out-of-domain (expected: "fallback").
* Candidate thresholds are compared on the CALIBRATION split only. The
  threshold with the best overall accuracy is chosen (ties -> the higher,
  safer threshold). The TEST split is then used once to report results, so
  the reported numbers are not tuned on the questions they are measured on.

Definitions
-----------
* Positive class = "in-domain question, the bot should answer".
  A prediction is positive when the bot answers (no fallback).
* In-domain precision = answered in-domain questions / all answered questions.
* In-domain recall    = answered in-domain questions / all in-domain questions.
* Fallback precision  = out-of-domain fallbacks / all fallbacks.
* Fallback recall     = out-of-domain fallbacks / all out-of-domain questions.
* Topic accuracy      = in-domain questions whose answer has an acceptable
                        topic (a fallback counts as wrong).
* Entry accuracy      = in-domain questions answered with an acceptable entry.
* Overall accuracy    = in-domain answered with an acceptable entry, or
                        out-of-domain given a fallback, over all questions.
* Confidently wrong   = answers given with the wrong entry (including any
                        answer to an out-of-domain question).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import config
from nlp.chatbot import PlantCareChatbot

EVAL_DIR = config.BASE_DIR / "evaluation"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
REPORT_DIR = EVAL_DIR / "reports"
CANDIDATE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def load_dataset(path: Path = DATASET_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for split in ("calibration", "test"):
        for item in data[split]:
            if item["expected"] not in ("answer", "fallback"):
                raise ValueError(f"Bad 'expected' value in {split}: {item}")
            if item["expected"] == "answer" and not item["expected_ids"]:
                raise ValueError(f"In-domain item needs expected_ids: {item}")
    return data


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def evaluate_split(bot: PlantCareChatbot, items: list[dict], threshold: float) -> dict:
    """Run every question at one threshold and compute the metrics."""
    rows = []
    for item in items:
        result = bot.answer(item["query"], low_threshold=threshold)
        in_domain = item["expected"] == "answer"
        acceptable_topics = {bot.entries_by_id[i]["topic"] for i in item["expected_ids"]}
        answered = not result.fallback
        entry_ok = answered and result.matched_id in item["expected_ids"]
        topic_ok = answered and result.topic in acceptable_topics
        rows.append({
            "query": item["query"],
            "expected": item["expected"],
            "expected_ids": item["expected_ids"],
            "answered": answered,
            "matched_id": result.matched_id,
            "topic": result.topic,
            "similarity": round(result.similarity, 3),
            "fallback_reason": result.fallback_reason,
            "entry_correct": entry_ok if in_domain else None,
            "topic_correct": topic_ok if in_domain else None,
            "correct": entry_ok if in_domain else not answered,
        })

    in_rows = [r for r in rows if r["expected"] == "answer"]
    out_rows = [r for r in rows if r["expected"] == "fallback"]
    true_pos = sum(r["answered"] for r in in_rows)
    false_pos = sum(r["answered"] for r in out_rows)
    true_fallback = sum(not r["answered"] for r in out_rows)
    false_fallback = sum(not r["answered"] for r in in_rows)
    precision = _ratio(true_pos, true_pos + false_pos)
    recall = _ratio(true_pos, len(in_rows))
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else 0.0

    return {
        "threshold": threshold,
        "counts": {"total": len(rows), "in_domain": len(in_rows), "out_of_domain": len(out_rows)},
        "metrics": {
            "overall_accuracy": _ratio(sum(r["correct"] for r in rows), len(rows)),
            "topic_accuracy_in_domain": _ratio(sum(r["topic_correct"] for r in in_rows), len(in_rows)),
            "entry_accuracy_in_domain": _ratio(sum(r["entry_correct"] for r in in_rows), len(in_rows)),
            "in_domain_precision": precision,
            "in_domain_recall": recall,
            "in_domain_f1": f1,
            "fallback_precision": _ratio(true_fallback, true_fallback + false_fallback),
            "fallback_recall": _ratio(true_fallback, len(out_rows)),
            "confidently_wrong": sum(r["answered"] and not r["correct"] for r in rows),
        },
        "rows": rows,
    }


def choose_threshold(sweep: list[dict]) -> float:
    """Best calibration overall accuracy; ties go to the higher (safer) threshold."""
    best = max(sweep, key=lambda s: (s["metrics"]["overall_accuracy"], s["threshold"]))
    return best["threshold"]


def run_evaluation(bot: PlantCareChatbot | None = None) -> dict:
    bot = bot or PlantCareChatbot()
    data = load_dataset()
    calibration_sweep = [evaluate_split(bot, data["calibration"], t) for t in CANDIDATE_THRESHOLDS]
    chosen = choose_threshold(calibration_sweep)
    test_result = evaluate_split(bot, data["test"], chosen)
    # Shown for transparency only - NOT used to pick the threshold.
    test_sweep = [evaluate_split(bot, data["test"], t) for t in CANDIDATE_THRESHOLDS]
    return {
        "generated_on": date.today().isoformat(),
        "knowledge_base_entries": len(bot.entries),
        "configured_threshold": config.LOW_CONFIDENCE_THRESHOLD,
        "chosen_threshold": chosen,
        "calibration_sweep": [{"threshold": s["threshold"], **s["metrics"]} for s in calibration_sweep],
        "test_sweep_for_information": [{"threshold": s["threshold"], **s["metrics"]} for s in test_sweep],
        "test_result": test_result,
    }


def _metric_table(sweep: list[dict]) -> list[str]:
    lines = [
        "| Threshold | Overall acc. | Topic acc. | Entry acc. | In-domain P | In-domain R | F1 | Fallback P | Fallback R | Confidently wrong |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sweep:
        lines.append(
            f"| {s['threshold']:.2f} | {s['overall_accuracy']} | {s['topic_accuracy_in_domain']} | "
            f"{s['entry_accuracy_in_domain']} | {s['in_domain_precision']} | {s['in_domain_recall']} | "
            f"{s['in_domain_f1']} | {s['fallback_precision']} | {s['fallback_recall']} | {s['confidently_wrong']} |"
        )
    return lines


def to_markdown(report: dict) -> str:
    test = report["test_result"]
    m = test["metrics"]
    lines = [
        "# GreenBuddy retrieval evaluation report",
        "",
        f"Generated by `python3 -m nlp.evaluator` on {report['generated_on']} "
        f"({report['knowledge_base_entries']} knowledge-base entries).",
        "",
        "## Chosen threshold",
        "",
        f"* Chosen on the calibration split: **{report['chosen_threshold']:.2f}**",
        f"* Value currently in `config.py`: **{report['configured_threshold']:.2f}**",
        "",
        "## Held-out test split results",
        "",
        f"{test['counts']['total']} questions ({test['counts']['in_domain']} in-domain, "
        f"{test['counts']['out_of_domain']} out-of-domain), threshold {test['threshold']:.2f}.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    labels = {
        "overall_accuracy": "Overall accuracy",
        "topic_accuracy_in_domain": "Top-1 topic accuracy (in-domain)",
        "entry_accuracy_in_domain": "Top-1 entry accuracy (in-domain)",
        "in_domain_precision": "In-domain precision",
        "in_domain_recall": "In-domain recall",
        "in_domain_f1": "In-domain F1",
        "fallback_precision": "Fallback precision",
        "fallback_recall": "Fallback recall",
        "confidently_wrong": "Confidently wrong answers (count)",
    }
    lines += [f"| {labels[key]} | {value} |" for key, value in m.items()]
    lines += ["", "## Calibration sweep (used to choose the threshold)", ""]
    lines += _metric_table(report["calibration_sweep"])
    lines += ["", "## Test sweep (for information only, not used for selection)", ""]
    lines += _metric_table(report["test_sweep_for_information"])
    failures = [r for r in test["rows"] if not r["correct"]]
    lines += ["", "## Test-split failures", ""]
    if failures:
        lines += ["| Question | Expected | Got | Similarity |", "|---|---|---|---|"]
        for r in failures:
            expected = ", ".join(r["expected_ids"]) if r["expected_ids"] else "fallback"
            got = r["matched_id"] or f"fallback ({r['fallback_reason']})"
            lines.append(f"| {r['query']} | {expected} | {got} | {r['similarity']} |")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Caveats",
        "",
        "* The dataset is small (80 questions), so one question changes a metric by 2.5-3.6 percentage points.",
        "* The questions were written by the project author; real users will phrase things differently.",
        "* The domain check and two question templates were improved once after an earlier run that included "
        "this test split, so the test numbers are somewhat optimistic.",
        "* Similarity scores are cosine similarities between TF-IDF vectors, not probabilities.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    report = run_evaluation()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (REPORT_DIR / "evaluation_report.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    m = report["test_result"]["metrics"]
    print(f"Chosen threshold (calibration split): {report['chosen_threshold']:.2f}")
    print(f"Held-out test: overall accuracy {m['overall_accuracy']}, topic accuracy {m['topic_accuracy_in_domain']}, "
          f"in-domain P/R {m['in_domain_precision']}/{m['in_domain_recall']}, "
          f"fallback recall {m['fallback_recall']}, confidently wrong {m['confidently_wrong']}")
    print(f"Reports written to {REPORT_DIR.relative_to(config.BASE_DIR)}/")
    if abs(report["chosen_threshold"] - config.LOW_CONFIDENCE_THRESHOLD) > 1e-9:
        print(f"NOTE: config.LOW_CONFIDENCE_THRESHOLD is {config.LOW_CONFIDENCE_THRESHOLD}; "
              f"the calibration suggests {report['chosen_threshold']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
