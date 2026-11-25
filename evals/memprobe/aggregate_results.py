"""Compute native-run metrics and evidence indexes as JSON; no document generation."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
import statistics

from common import HERE, RUN, dump, load_episodes, product_hashes, sha256, utcnow
from execution import read_plan, validate_baseline, scoped_episodes
from scoring import build_episode_meta, compute_baseline_metrics

TYPES = {"current_value": "当前事实", "previous_value": "历史事实", "change_detection": "变化检测",
         "conflict_value": "冲突值", "source": "来源识别", "temporal": "时间关系"}
FINAL_LABELS = {"correct", "partial", "incorrect", "execution_error"}


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo) if lo != hi else xs[lo]


def observed_usage(calls):
    calls = [c for c in calls if not c.get("offline_mock_transport")]
    return {"requests": len(calls), "completed_requests": sum(c.get("status") == "completed" for c in calls),
        "inflight_requests": sum(c.get("status") == "started" for c in calls),
        "missing_usage_requests": sum(c.get("status") != "started" and not (c.get("usage") or {}).get("input_tokens") for c in calls),
        **{k: sum((c.get("usage") or {}).get(k, 0) for c in calls)
           for k in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")}}


def evaluated(row):
    return row["match"] in FINAL_LABELS and row["execution_status"] in {"completed", "product_failure"}


def distribution(rows):
    labels = Counter(r["match"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {"planned": len(rows), "evaluated": sum(evaluated(r) for r in rows),
            "correct": correct, "partial": labels["partial"], "incorrect": labels["incorrect"],
            "execution_error": labels["execution_error"], "not_scored": labels["not_scored"],
            "accuracy": correct / len(rows) if rows else None}


def question_key(row):
    return row["episode_id"], row["probe_index"]


def unique_rows(rows):
    result = {}
    for row in rows:
        key = question_key(row)
        if key in result:
            raise RuntimeError("Duplicate question result: " + str(key))
        result[key] = row
    return result


def compare_rows(current, old):
    before = unique_rows(old)
    unique_rows(current)
    pairs = []
    for row in current:
        prior = before.get(question_key(row))
        if prior is None:
            raise RuntimeError("Historical question missing: " + str(question_key(row)))
        if any(row[k] != prior[k] for k in ("question", "gold", "probe_type")):
            raise RuntimeError("Question/gold mismatch; refusing version comparison")
        if not evaluated(row) or not evaluated(prior):
            continue
        pairs.append({"episode_id": row["episode_id"], "probe_index": row["probe_index"],
                      "group_id": row["group_id"], "probe_type": row["probe_type"],
                      "before": prior["match"], "after": row["match"],
                      "correct_before": prior["correct"], "correct_after": row["correct"],
                      "input_before": prior["usage"]["input_tokens"], "input_after": row["usage"]["input_tokens"],
                      "complete_usage": all(r["usage"]["requests"] > 0 and
                          not r["usage"]["missing_usage_requests"] and not r["usage"]["inflight_requests"] for r in (prior, row))})
    improved = [p for p in pairs if not p["correct_before"] and p["correct_after"]]
    regressed = [p for p in pairs if p["correct_before"] and not p["correct_after"]]
    cost_pairs = [p for p in pairs if p["complete_usage"]]
    old_input, new_input = (sum(p[k] for p in cost_pairs) for k in ("input_before", "input_after"))
    return {"paired_questions": len(pairs), "improved": len(improved), "regressed": len(regressed),
            "net_correct": len(improved) - len(regressed),
            "accuracy_delta": (len(improved) - len(regressed)) / len(pairs) if pairs else None,
            "transitions": dict(Counter(p["before"] + " -> " + p["after"] for p in pairs)),
            "by_type": {t: {"paired": len(pp), "improved": sum(not p["correct_before"] and p["correct_after"] for p in pp),
                            "regressed": sum(p["correct_before"] and not p["correct_after"] for p in pp),
                            "accuracy_before": sum(p["correct_before"] for p in pp) / len(pp) if pp else None,
                            "accuracy_after": sum(p["correct_after"] for p in pp) / len(pp) if pp else None}
                        for t in TYPES for pp in [[p for p in pairs if p["probe_type"] == t]]},
            "usage_complete_pairs": len(cost_pairs), "input_before": old_input, "input_after": new_input,
            "input_change": new_input / old_input - 1 if old_input else None,
            "confidence_intervals": version_bootstrap(pairs),
            "pairs": pairs, "regressions": regressed}


def version_bootstrap(pairs, iterations=2000):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["group_id"]].append(pair)
    if len(groups) < 2:
        return None
    names, rng = sorted(groups), random.Random(20261121)
    scores = []
    for _ in range(iterations):
        sample = [p for _ in names for p in groups[rng.choice(names)]]
        scores.append(sum(int(p["correct_after"]) - int(p["correct_before"]) for p in sample) / len(sample))
    return {"method": "paired group-cluster percentile bootstrap", "groups": len(names),
            "iterations": iterations, "seed": 20261121,
            "accuracy_delta": [percentile(scores, .025), percentile(scores, .975)],
            "scope": "One historical and one new memory construction; not a randomized ablation or repeatability estimate"}


def aggregate_results(directory, episodes, *, baseline_rows=None, check_baseline=True):
    directory = Path(directory)
    episodes = scoped_episodes(episodes, directory)
    plan = read_plan(directory)
    if plan["mode"] != "single":
        raise ValueError("JSON aggregation requires the native execution plan")
    config = read(directory / "formal_config.json", read(directory / "product_baseline.json", {}))
    progress = read(directory / "progress.json", {"status": "NOT_STARTED", "episodes": {}, "blockers": []})
    calls = []
    for path in sorted((directory / "episodes").glob("*/**/model_calls/*.json")):
        call = read(path)
        if call.get("offline_mock_transport"):
            raise RuntimeError("Mock calls cannot be reported as formal results")
        call["artifact"] = path.relative_to(directory).as_posix()
        calls.append(call)
    if len({c["id"] for c in calls}) != len(calls):
        raise RuntimeError("Duplicate model call IDs")
    answer_calls = defaultdict(list)
    for call in calls:
        if call.get("phase") == "answer":
            answer_calls[(call.get("episode_id"), call.get("unit_id"), call.get("arm"))].append(call)
    wrong_arms = [p for p in (directory / "episodes").glob("*/answers/q*/*") if p.is_dir() and p.name != "native"]
    if wrong_arms:
        raise RuntimeError("Paired results found in native-only run")
    rows, official, verification_tasks = [], {}, 0
    for ep in episodes:
        for qi, probe in enumerate(ep["probes"]):
            path = directory / "episodes" / ep["episode_id"] / "answers" / f"q{qi}" / "native"
            result, score = read(path / "trial/result.json", {}), read(path / "score.json", {})
            label = score.get("match", "not_scored")
            if label not in FINAL_LABELS | {"not_scored"}:
                raise RuntimeError("Unrecognized score label: " + label)
            if score:
                if (score.get("episode_id"), score.get("probe_index"), score.get("arm")) != (ep["episode_id"], qi, "native"):
                    raise RuntimeError("Score identity mismatch")
                official.setdefault(ep["episode_id"], {"answers": []})["answers"].append(score)
            # Keep costs from archived no-answer environment attempts. Ledger
            # records override duplicate result summaries by immutable call ID.
            archived_paths = sorted((directory / "episodes" / ep["episode_id"] /
                "environment_attempts").glob(f"q{qi}-native-*/trial/result.json"))
            archived = [read(p) for p in archived_paths]
            attempts = [*archived, result]
            task_call_map = {c["id"]: c for a in attempts for c in a.get("calls", [])}
            task_call_map.update({c["id"]: c for c in answer_calls[(ep["episode_id"], f"q{qi}", "native")]})
            task_calls = list(task_call_map.values())
            events = [e for a in attempts for e in a.get("events", [])]
            archived_seconds = sum(a.get("seconds") or 0 for a in archived)
            seconds = (result["seconds"] + archived_seconds if result.get("seconds") is not None
                       else archived_seconds if archived else None)
            verification_tools = any(e.get("origin") == "memory_verification" and e.get("type") == "tool_call" for e in events)
            verification = any(e.get("origin") == "memory_verification" for e in events) or any(
                m.get("role") == "system" and str(m.get("content", "")).startswith("Memory verification (runtime)")
                for e in events for m in e.get("request", {}).get("messages", []))
            verification_tasks += verification
            observations = [t for a in attempts for t in a.get("tool_observations", [])]
            rows.append({"episode_id": ep["episode_id"], "group_id": ep["group_id"], "paradigm": ep["paradigm"],
                "probe_index": qi, "probe_type": probe["probe_type"], "question": probe["question"], "gold": probe["gold_answer"],
                "arm": "native", "execution_status": result.get("status", "not_executed"),
                "match": label, "correct": label == "correct" and result.get("status") == "completed", "score": score,
                "answer": result.get("answer"), "usage": observed_usage(task_calls), "seconds": seconds,
                "final_attempt_seconds": result.get("seconds"), "archived_attempt_seconds": archived_seconds,
                "archived_attempt_count": len(archived),
                "initial_candidate_ids": [m["id"] for m in result.get("candidates", [])],
                "memory_notices": result.get("memory_notices", []), "tool_calls": observations,
                "memory_read_count": sum(t["name"] == "memory_read" for t in observations),
                "runtime_verification": verification,
                "runtime_verification_tools": verification_tools,
                "runtime_verification_prompt_only": verification and not verification_tools,
                "evidence": {"result": (path / "trial/result.json").relative_to(directory).as_posix(),
                             "archived_results": [p.relative_to(directory).as_posix() for p in archived_paths],
                             "score": (path / "score.json").relative_to(directory).as_posix(),
                             "snapshot": f"episodes/{ep['episode_id']}/snapshot.db"}})
    native = compute_baseline_metrics(official, build_episode_meta(episodes)) if official else {}
    finished = [r for r in rows if evaluated(r)]
    executed = [r for r in rows if r["execution_status"] != "not_executed"]
    cost_complete = [r for r in finished if r["usage"]["requests"] and not r["usage"]["missing_usage_requests"]
                     and not r["usage"]["inflight_requests"]]
    times = [r["seconds"] for r in executed if r["seconds"] is not None]
    summary = {**distribution(rows), "native": native,
        "completed": sum(r["execution_status"] == "completed" for r in rows),
        "completion_rate": sum(r["execution_status"] == "completed" for r in rows) / len(rows) if rows else None,
        "status_counts": dict(Counter(r["execution_status"] for r in rows)),
        "by_type": {t: distribution([r for r in rows if r["probe_type"] == t]) for t in TYPES},
        "by_paradigm": {t: distribution([r for r in rows if r["paradigm"] == t]) for t in sorted({r["paradigm"] for r in rows})},
        "complete_usage_tasks": len(cost_complete),
        "mean_input_tokens_complete_usage": statistics.mean(r["usage"]["input_tokens"] for r in cost_complete) if cost_complete else None,
        "mean_output_tokens_complete_usage": statistics.mean(r["usage"]["output_tokens"] for r in cost_complete) if cost_complete else None,
        "mean_model_calls_executed": statistics.mean(r["usage"]["requests"] for r in executed) if executed else None,
        "seconds": {"mean": statistics.mean(times) if times else None, "p50": percentile(times, .5), "p95": percentile(times, .95)},
        "input_tokens": {"p50": percentile([r["usage"]["input_tokens"] for r in cost_complete], .5),
                         "p95": percentile([r["usage"]["input_tokens"] for r in cost_complete], .95)},
        "tool_calls": dict(Counter(t["name"] for r in rows for t in r["tool_calls"]))}
    comparison, compatibility = None, {}
    if baseline_rows is None and check_baseline and plan.get("baseline"):
        baseline_dir = validate_baseline(plan)
        old_config = read(baseline_dir / "formal_config.json")
        old_rows = read(baseline_dir / "reports/per_question.json")
        baseline_rows = [r for r in old_rows if r["arm"] == plan["baseline"]["arm"]]
        for key in ("dataset_revision", "dataset_files"):
            if config.get(key) is not None and config[key] != old_config[key]:
                raise RuntimeError("Historical dataset is not comparable: " + key)
        current_scorer = sha256(HERE / "scoring.py")
        if current_scorer != old_config["adapter_sources"]["scoring.py"]:
            raise RuntimeError("Scoring adapter differs from historical baseline")
        compatibility = {k: {"same": config.get(k) == old_config.get(k),
                             "before": old_config.get(k), "after": config.get(k)}
                         for k in ("effective_models", "roles", "answer_budget", "synthetic_clock", "timeout_policy", "requested_sampling")}
    if baseline_rows is not None:
        comparison = compare_rows(rows, baseline_rows)
        comparison["configuration_comparison"] = compatibility
        comparison["interpretation"] = "Historical regression comparison on previously inspected data; no isolated causal or unseen-data claim"
    phase_calls = defaultdict(list)
    for call in calls:
        phase_calls[call["phase"]].append(call)
    phase_usage = {phase: {**observed_usage(cc), "sum_call_seconds": sum(c.get("seconds", 0) for c in cc)} for phase, cc in phase_calls.items()}
    memories = [m for p in (directory / "episodes").glob("*/stored_memories.json") for m in read(p)]
    ingestion_results = [read(p) for p in (directory / "episodes").glob("*/ingestion/turns/*/result.json")]
    memory_failures = [{k: r.get(k) for k in ("episode_id", "unit_id", "status", "error_type", "error", "failure_scope")}
                       for r in ingestion_results if r.get("status") == "product_failure"]
    stage_timing = {"ingestion_task_seconds_sum": sum(r.get("seconds", 0) for r in ingestion_results),
                    "ingestion_units_recorded": len(ingestion_results),
                    "answer_task_seconds_sum": sum(times),
                    "note": "Sums over tasks; concurrent work overlaps, so these are not wall-clock durations"}
    mechanisms = {"runtime_verification_tasks": verification_tasks,
        "runtime_verification_tool_tasks": sum(r["runtime_verification_tools"] for r in rows),
        "runtime_verification_prompt_only_tasks": sum(r["runtime_verification_prompt_only"] for r in rows),
        "extraction_review_requests": sum("Review the original extraction against the sources." in str(c.get("request", {}).get("messages", [{}])[0].get("content", "")) for c in calls),
        "reextraction_requests": sum(str(c.get("request", {}).get("messages", [{}])[0].get("content", "")).startswith("Repair one long-term memory extraction") for c in calls),
        "records_with_reassessment": sum(bool(m.get("auto_reassessment_count")) for m in memories),
        "reassessment_final_states": dict(Counter(m.get("decision_state", "unknown") for m in memories if m.get("auto_reassessment_count"))),
        "records_with_applied_extraction_correction": sum(bool(m.get("extraction_correction")) for m in memories),
        "note": "Execution observations, not semantic repair success rates or causal attribution"}
    synthetic_calls = [read(p) for p in (directory / "synthetic").glob("*/**/model_calls/*.json")]
    start, end = progress.get("started_at"), progress.get("updated_at")
    scope = config.get("execution_subset")
    complete = (progress["status"] in ({"PILOT_COMPLETE"} if scope else {"COMPLETE"}) and len(finished) == len(rows) and not progress.get("blockers")
                and sum(s.get("status") == "complete" for s in progress.get("episodes", {}).values()) == len(episodes))
    metrics = {"generated_at": utcnow(), "mode": "single", "complete": complete, "execution_subset": scope, "run_status": progress["status"],
        "summary": summary, "historical_comparison": comparison, "mechanisms": mechanisms,
        "all_formal_usage": observed_usage(calls), "usage_by_phase": phase_usage,
        "synthetic_preflight_usage": observed_usage(synthetic_calls),
        "scoring_usage": {"model_calls": 0, "tokens": 0, "method": "pinned native rules"},
        "stage_timing": stage_timing,
        "memory_processing_failures": memory_failures,
        "wall_seconds_including_pauses": (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None,
        "product_sources_unchanged": product_hashes() == config.get("product_sources"), "blockers": progress.get("blockers", [])}
    report_dir = directory / "reports"
    dump(report_dir / "metrics.json", metrics)
    dump(report_dir / "per_question.json", rows)
    dump(report_dir / "official_format_results.json", {"native": official})
    dump(report_dir / "failure_analysis.json", [r for r in rows if evaluated(r) and not r["correct"]])
    dump(report_dir / "model_call_index.json", [{k: c.get(k) for k in ("id", "phase", "episode_id", "unit_id", "arm", "status", "usage", "seconds", "artifact", "error")} for c in calls])
    dump(report_dir / "version_comparison.json", comparison)
    dump(report_dir / "regressions.json", comparison["regressions"] if comparison else [])
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in {".json", ".db"}
             and "settings" not in p.parts and "workspace" not in p.parts and "reports" not in p.parts]
    dump(report_dir / "delivery_manifest.json", {"files": {p.relative_to(directory).as_posix(): sha256(p) for p in files},
         "credentials_included": False, "mode": "single"})
    return metrics


def main():
    metrics = aggregate_results(RUN, load_episodes())
    print({"status": metrics["run_status"], "complete": metrics["complete"],
           "evaluated": metrics["summary"]["evaluated"], "planned": metrics["summary"]["planned"]})


if __name__ == "__main__":
    main()
