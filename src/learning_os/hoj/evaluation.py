"""Offline evidence metrics, not a production answer/relevance classifier.

Ground truth comes from fixed, human-authored document/section/span labels.
A returned source alone is not a match. No embedder, model key or IO here.
"""

from __future__ import annotations

import math


def relevant(hit: dict, label: dict) -> bool:
    return (hit["document"] == label["document"]
            and hit["headingPath"].split(" / ")[-1] == label["heading"]
            and all(span in hit["excerpt"] for span in label["evidence"]))


def measure(case: dict, hits: list[dict]) -> dict:
    labels = case["relevant"]
    ranks = [next((i for i, hit in enumerate(hits, 1) if relevant(hit, label)), None)
             for label in labels]
    row = {"id": case["id"], "split": case["split"], "query": case["query"],
           "answerable": bool(labels), "evidenceGroups": len(labels), "ranks": ranks,
           "category": case.get("category", "answerable"), "reason": case.get("reason"),
           "documents": case.get("documents"), "hits": hits}
    if labels:
        row["recall"] = {str(k): sum(r is not None and r <= k for r in ranks) / len(labels)
                         for k in (1, 3, 5)}
        row["successAt3"] = all(r is not None and r <= 3 for r in ranks)
        found = [r for r in ranks if r is not None and r <= 5]
        row["reciprocalRankAt5"] = 1 / min(found) if found else 0.0
    return row


def summarize(rows: list[dict]) -> dict:
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    total = sum(r["evidenceGroups"] for r in positive)
    return {"queries": len(rows), "answerable": len(positive), "unanswerable": len(negative),
            "evidenceGroups": total,
            "evidenceRecall": {str(k): (sum(sum(rank is not None and rank <= k for rank in r["ranks"])
                                              for r in positive) / total if total else None)
                               for k in (1, 3, 5)},
            "successAt3": (sum(r["successAt3"] for r in positive) / len(positive) if positive else None),
            "mrrAt5": (sum(r["reciprocalRankAt5"] for r in positive) / len(positive) if positive else None),
            "unanswerableWithCandidates": sum(bool(r["hits"]) for r in negative),
            "unanswerableCandidateRate": (sum(bool(r["hits"]) for r in negative) / len(negative)
                                            if negative else None)}


def operating_point(rows: list[dict], threshold: float) -> dict:
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    total = sum(r["evidenceGroups"] for r in positive)
    kept = sum(sum(rank is not None and r["hits"][rank - 1]["score"] >= threshold for rank in r["ranks"])
               for r in positive)
    accepted = sum(any(h["score"] >= threshold for h in r["hits"]) for r in negative)
    categories = sorted({r["category"] for r in negative})
    return {"threshold": threshold, "evidenceRecallAt5": kept / total if total else None,
            "unanswerableFalseAcceptRate": accepted / len(negative) if negative else None,
            "unanswerableAccepted": accepted, "unanswerableCount": len(negative),
            "byCategory": {c: {"count": sum(r["category"] == c for r in negative),
                               "accepted": sum(r["category"] == c and any(h["score"] >= threshold for h in r["hits"])
                                               for r in negative)} for c in categories}}


def threshold_study(rows: list[dict]) -> dict:
    calibration = [r for r in rows if r["split"] == "calibration"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    if any(not any(r["answerable"] == answerable for r in group)
           for group in (calibration, heldout) for answerable in (True, False)):
        raise ValueError("Each fixed split needs supported and unsupported questions")
    # Observed boundary points suffice for >= comparisons; use nextafter to
    # evaluate excluding a score without rounding or tuning on heldout.
    scores = {h["score"] for r in calibration for h in r["hits"]}
    candidates = sorted(scores | {math.nextafter(s, math.inf) for s in scores} | {-1.0, 1.000001})
    curve = [operating_point(calibration, value) for value in candidates]
    feasible = [p for p in curve if p["evidenceRecallAt5"] >= 0.90
                and p["unanswerableFalseAcceptRate"] <= 0.05]
    chosen = feasible[0] if feasible else None
    evaluated = operating_point(heldout, chosen["threshold"]) if chosen else None
    return {"selectionSplit": "calibration", "selectionRule": "lowest threshold with recall>=0.90 and falseAccept<=0.05",
            "feasibleOnCalibration": bool(chosen), "calibrationChoice": chosen, "heldoutAtChosenThreshold": evaluated,
            "heldoutMeetsRule": bool(evaluated and evaluated["evidenceRecallAt5"] >= 0.90
                                     and evaluated["unanswerableFalseAcceptRate"] <= 0.05),
            "calibrationCurve": curve,
            "warning": "Similarity gating is not an entailment check or proof of answer correctness; no runtime threshold is changed."}
