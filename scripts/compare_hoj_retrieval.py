"""Compare same-data/same-implementation dense and hybrid evidence reports.

This is a deployment gate for retrieval regression, not an answer evaluator.
It never edits labels, changes a threshold, runs a model or accesses HOJ.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def compare(dense: dict, hybrid: dict) -> dict:
    for result, mode in ((dense, "dense"), (hybrid, "hybrid")):
        if not result.get("executionComplete") or not result.get("integrityPassed") or not result.get("cleaned"):
            raise ValueError("A comparison requires complete, cleaned, integrity-checked runs")
        if result.get("retrievalMode") != mode or result.get("externalModelCalls") != 0 or result.get("liveDatabaseOrIndexAccessed") is not False:
            raise ValueError("Invalid mode or evaluation scope")
    for key in ("dataset", "implementationHashes", "modelConfiguration", "runtimeVersions", "indexedChunks", "chunksByDocument"):
        if dense.get(key) != hybrid.get(key):
            raise ValueError("Cannot compare different " + key)
    if [c["id"] for c in dense["cases"]] != [c["id"] for c in hybrid["cases"]]:
        raise ValueError("Query sets differ")
    regressions, improvements = [], []
    for before, after in zip(dense["cases"], hybrid["cases"], strict=True):
        if before.get("answerable") and before.get("ranks") != after.get("ranks"):
            target = regressions if after["recall"]["3"] < before["recall"]["3"] else improvements
            target.append({"id": before["id"], "split": before["split"], "before": before["ranks"], "after": after["ranks"]})
    gates = []
    for split in ("all", "heldout"):
        for metric in ("1", "3", "5"):
            before, after = dense[split]["evidenceRecall"][metric], hybrid[split]["evidenceRecall"][metric]
            gates.append({"split": split, "metric": "evidenceRecall@" + metric, "before": before, "after": after, "passed": after >= before})
        for metric in ("successAt3", "mrrAt5"):
            before, after = dense[split][metric], hybrid[split][metric]
            gates.append({"split": split, "metric": metric, "before": before, "after": after, "passed": after >= before})
    timing = {}
    for mode, data in (("dense", dense), ("hybrid", hybrid)):
        values = sorted(row["elapsedMs"] for row in data["cases"])
        timing[mode] = {"medianMs": statistics.median(values), "maxMs": max(values), "p95Ms": values[min(len(values)-1, int(len(values)*.95))]}
    budget = max(timing["dense"]["p95Ms"] * 1.5, timing["dense"]["p95Ms"] + 250)
    gates.append({"split": "all", "metric": "p95Ms", "before": timing["dense"]["p95Ms"],
                  "after": timing["hybrid"]["p95Ms"], "budget": budget, "passed": timing["hybrid"]["p95Ms"] <= budget})
    return {"dataset": dense["dataset"], "gatePassed": all(g["passed"] for g in gates), "gates": gates,
            "timing": timing, "changedRanks": {"lostEvidenceAt3": regressions, "otherChanges": improvements},
            "unsupportedCandidates": {mode: data["all"]["unanswerableWithCandidates"] for mode, data in (("dense", dense), ("hybrid", hybrid))},
            "answerQualityValidated": False, "runtimeThresholdChanged": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--hybrid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.dense.read_text(encoding="utf-8")), json.loads(args.hybrid.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gatePassed", "gates", "timing", "unsupportedCandidates")}, indent=2))
    return 0 if result["gatePassed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
