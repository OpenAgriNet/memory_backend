"""Prepare blind A/B reading bundles and apply direct assistant review to the CSV."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

WORK = Path(os.getenv("AMUL_BACKTEST_WORK", str(Path(__file__).resolve().parents[2] / "amul-memory-backtest/20260909-20-farmers")))


def arms(case_id):
    return ("on", "off") if hashlib.sha256((case_id + ":blind-v1").encode()).digest()[0] % 2 else ("off", "on")


def show(start, end):
    ids = json.loads((WORK / "case-index.json").read_text())[start:end]
    for case_id in ids:
        path = WORK / "results" / (case_id + ".json")
        if not path.exists():
            print(case_id, "not ready")
            continue
        row = json.loads(path.read_text())
        if not row.get("on", {}).get("answer") or not row.get("off", {}).get("answer"):
            print(case_id, "incomplete", row.get("on", {}).get("error"), row.get("off", {}).get("error"))
            continue
        a, b = arms(case_id)
        print(json.dumps({"case_id": case_id, "query": row["query"], "query_en": row["query_en"],
                          "A": row[a]["answer"], "B": "[Identical to A]" if row[a]["answer"] == row[b]["answer"] else row[b]["answer"],
                          "identical": row[a]["answer"] == row[b]["answer"],
                          "limitations": sorted(set(row[a]["limitations"] + row[b]["limitations"]))}, ensure_ascii=False))


def apply():
    reviews = json.loads((WORK / "reviews.json").read_text())
    fields = ["case_id", "farmer", "cutoff", "query_ts", "query", "query_en", "answer_without_memory",
              "answer_with_memory", "memory_improvement_score", "review_reason", "review_confidence",
              "without_memory_seconds", "with_memory_seconds", "memory_tool_calls", "memory_context_chars",
              "replay_limitations", "generation_error", "issue_classification", "causal_attribution"]
    rows = []
    for case_id in json.loads((WORK / "case-index.json").read_text()):
        path = WORK / "results" / (case_id + ".json")
        row = json.loads(path.read_text())
        review = reviews[case_id]
        assert all(row.get(arm, {}).get("answer") and not row[arm].get("error") for arm in ("off", "on")), case_id
        assert review["reason"].strip() and review["confidence"] in ("high", "medium", "low"), case_id
        preferred = review["preferred"]
        magnitude = review["magnitude"]
        assert preferred in ("A", "B", "tie") and magnitude in (0, 1, 2)
        assert (preferred == "tie") == (magnitude == 0)
        winner = None if preferred == "tie" else arms(case_id)[preferred == "B"]
        score = 0 if winner is None else magnitude if winner == "on" else -magnitude
        labels = {letter: "the answer with memory" if arm == "on" else "the answer without memory"
                  for letter, arm in zip(("A", "B"), arms(case_id))}
        reason = re.sub(r"\b[AB]\b", lambda match: labels[match[0]], review["reason"])
        reason = reason[0].upper() + reason[1:]
        row["review"] = {"score": score, "reason": reason, "confidence": review["confidence"],
                         "reviewer": "Codex assistant direct review", "blind_preference": preferred,
                         "issue_classification": review.get("issue_classification", ""),
                         "causal_attribution": review.get("causal_attribution", "")}
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
        off, on = row["off"], row["on"]
        rows.append({**{k: row[k] for k in fields[:6]}, "answer_without_memory": off.get("answer", ""),
                     "answer_with_memory": on.get("answer", ""), "memory_improvement_score": score,
                     "review_reason": reason, "review_confidence": review["confidence"],
                     "without_memory_seconds": round(off["seconds"], 3), "with_memory_seconds": round(on["seconds"], 3),
                     "memory_tool_calls": on["memory_calls"], "memory_context_chars": len(on["memory_context"]),
                     "replay_limitations": "; ".join(sorted(set(off["limitations"] + on["limitations"]))),
                     "generation_error": off.get("error", "") or on.get("error", ""),
                     "issue_classification": review.get("issue_classification", ""),
                     "causal_attribution": review.get("causal_attribution", "")})
    path = WORK / "memory_comparison.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)
    summary = {"cases": len(rows), "farmers": len({r["farmer"] for r in rows}),
               "improved": sum(r["memory_improvement_score"] > 0 for r in rows),
               "unchanged": sum(r["memory_improvement_score"] == 0 for r in rows),
               "worse": sum(r["memory_improvement_score"] < 0 for r in rows),
               "mean_score": statistics.mean(r["memory_improvement_score"] for r in rows),
               "median_seconds_off": statistics.median(r["without_memory_seconds"] for r in rows),
               "median_seconds_on": statistics.median(r["with_memory_seconds"] for r in rows),
               "memory_tool_use_cases": sum(r["memory_tool_calls"] > 0 for r in rows),
               "context_present_cases": sum(r["memory_context_chars"] > 0 for r in rows),
               "identical_answers": sum(r["answer_without_memory"] == r["answer_with_memory"] for r in rows),
               "fixture_limit_cases": sum(bool(r["replay_limitations"]) for r in rows),
               "missing_fixture_cases": sum("missing_fixture" in r["replay_limitations"] for r in rows),
               "p95_seconds_off": sorted(r["without_memory_seconds"] for r in rows)[int(.95 * len(rows))-1],
               "p95_seconds_on": sorted(r["with_memory_seconds"] for r in rows)[int(.95 * len(rows))-1],
               "generation_errors": sum(bool(r["generation_error"]) for r in rows)}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["show", "apply"])
    parser.add_argument("start", type=int, nargs="?", default=0)
    parser.add_argument("end", type=int, nargs="?", default=10)
    args = parser.parse_args()
    show(args.start, args.end) if args.action == "show" else apply()
