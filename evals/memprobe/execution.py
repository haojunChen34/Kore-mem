"""Execution layout, separate from product memory policy and benchmark scoring."""
from __future__ import annotations

import json
from pathlib import Path

from common import HERE, RUN, sha256

BASELINE_NAME = "2026-11-21-five-fixes-full56"
BASELINE_FILES = ("formal_config.json", "question_manifest.json", "reports/per_question.json", "reports/metrics.json")


def read_plan(directory=None):
    directory = Path(directory or RUN)
    formal = directory / "formal_config.json"
    if formal.exists():
        config = json.loads(formal.read_text(encoding="utf-8"))
        plan = config.get("evaluation", {"mode": "paired", "arms": ["A", "B"]})
    elif (directory / "execution_plan.json").exists():
        plan = json.loads((directory / "execution_plan.json").read_text(encoding="utf-8"))
    else:
        plan = new_plan()
    expected = ["native"] if plan.get("mode") == "single" else ["A", "B"]
    if plan.get("mode") not in {"single", "paired"} or plan.get("arms") != expected:
        raise ValueError("Invalid evaluation mode/arms")
    return plan


def new_plan():
    return {"mode": "single", "arms": ["native"], "candidate_source": "product_runtime",
            "snapshot_policy": "fresh_history_ingestion_then_independent_question_copies",
            "answer_learning": False, "pause_on_product_failure": True}


def arms(directory=None, question_index=0):
    selected = read_plan(directory)["arms"]
    return selected[::-1] if len(selected) == 2 and question_index % 2 else selected


def counts(episodes, selected):
    questions = sum(len(e["probes"]) for e in episodes)
    return {"episodes_planned": len(episodes), "questions_planned": questions,
            "answers_planned": questions * len(selected)}


def scoped_episodes(episodes, directory=None):
    """Apply an explicitly frozen subset; never infer scope from completed results."""
    directory = Path(directory or RUN)
    formal = directory / "formal_config.json"
    if formal.exists():
        scope = json.loads(formal.read_text(encoding="utf-8")).get("execution_subset")
    else:
        path = directory / "execution_scope.json"
        scope = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if not scope:
        return episodes
    ids = scope["episode_ids"]
    if not ids or len(ids) != len(set(ids)) or set(ids) - {e["episode_id"] for e in episodes}:
        raise ValueError("Invalid frozen episode subset")
    return [e for e in episodes if e["episode_id"] in ids]


def pilot_ids(episodes, single=True):
    if single:
        # One fixed first episode per paradigm; no answer or score is consulted.
        return [min(e["episode_id"] for e in episodes if e["paradigm"] == paradigm)
                for paradigm in sorted({e["paradigm"] for e in episodes})]
    groups = {"GRP-INT-01", "GRP-MIS-01", "GRP-CON-01", "GRP-REC-01"}
    return [e["episode_id"] for e in episodes if e["group_id"] in groups]


def validate_baseline(plan):
    baseline = plan.get("baseline")
    if not baseline:
        return None
    directory = (HERE / "runs" / baseline["run"]).resolve()
    if directory.parent != (HERE / "runs").resolve():
        raise ValueError("Baseline must be a run in the benchmark runs directory")
    for name, expected in baseline["files"].items():
        if sha256(directory / name) != expected:
            raise RuntimeError("Historical baseline changed: " + name)
    return directory
