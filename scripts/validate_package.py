"""Offline validation of the selected complete MemProbe answer round and its memories."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "evals/memprobe/vendor"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_files():
    excluded = {".git", ".venv", "__pycache__", ".local", "runs", "node_modules"}
    def visit(folder):
        for p in folder.iterdir():
            if p.name in excluded or p.name.endswith(".local.json"):
                continue
            if p.is_dir():
                yield from visit(p)
            else:
                yield p
    return sorted(visit(ROOT))


def expected_public_paths(episodes, vendor_manifest):
    """The compact publication layout; runtime output is never part of the package."""
    module = ast.parse((ROOT / "evals/memprobe/runner.py").read_text("utf-8"))
    frozen = [node for node in module.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "FROZEN_FILES"
                      for target in node.targets)]
    assert len(frozen) == 1
    adapter_files = ast.literal_eval(frozen[0].value)
    assert len(adapter_files) == len(set(adapter_files)) == 18
    assert all(Path(name).name == name and name.endswith(".py") for name in adapter_files)
    paths = {
        ".gitattributes", ".gitignore", "LICENSE", "README.md", "TEST_REPORT.md", "requirements.txt",
        "configs/memprobe.example.json", "scripts/validate_package.py",
        "reports/memprobe/configuration.json", "reports/memprobe/metrics.json",
        "reports/memprobe/per_question.json", "snapshots/README.md", "snapshots/manifest.json",
        "evals/memprobe/vendor/manifest.json",
    }
    paths.update("evals/memprobe/" + name for name in adapter_files)
    paths.update("evals/memprobe/vendor/" + name for name in vendor_manifest["files"])
    paths.update("snapshots/" + episode["episode_id"] + ".json" for episode in episodes)
    assert len(paths) == 96
    return paths


def stats(values):
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"n": len(values), "sum": sum(values), "mean": sum(values) / len(values),
            "p50": percentile(.5), "p95": percentile(.95), "min": min(values), "max": max(values)}


def assert_stats(values, expected):
    for key, actual in stats(values).items():
        assert math.isclose(actual, expected[key], rel_tol=1e-12, abs_tol=1e-8), (key, actual, expected[key])


def main():
    manifest = read(VENDOR / "manifest.json")
    assert manifest["revision"] == "9068d8d9a2a47268fe6f7601b99a56a283c37be5"
    for name, expected in manifest["files"].items():
        assert sha(VENDOR / name) == expected, name
    episodes = [json.loads(s) for s in (VENDOR / "data/suite56/episodes.jsonl").read_text("utf-8").splitlines() if s.strip()]
    probes = {(e["episode_id"], i): (p, e["latent_spec"]) for e in episodes for i, p in enumerate(e["probes"])}
    assert len(episodes) == 56 and len(probes) == 224
    assert Counter(p[0]["probe_type"] for p in probes.values()) == {
        "current_value": 56, "previous_value": 28, "change_detection": 56,
        "conflict_value": 28, "source": 34, "temporal": 22}

    sys.path.insert(0, str(VENDOR / "src/eval"))
    sys.path.insert(0, str(VENDOR / "src/analysis"))
    from protocol import score_answer
    from compute_metrics import build_episode_meta, compute_baseline_metrics

    rows = read(ROOT / "reports/memprobe/per_question.json")
    metrics = read(ROOT / "reports/memprobe/metrics.json")
    config = read(ROOT / "reports/memprobe/configuration.json")
    assert len(rows) == 224
    assert {(r["episode_id"], r["probe_index"]) for r in rows} == set(probes)
    assert {r["source_run"] for r in rows} == {metrics["source_run"]} == {config["answer_source_run"]}
    assert metrics["source_run"] == "2026-11-24-frozen-conditional85"
    assert all(r["execution_status"] == "completed" and r["attempt_count"] == 1 for r in rows)
    by_question = {(r["episode_id"], r["probe_index"]): r for r in rows}
    official = {}
    for e in episodes:
        answers = []
        for i, probe in enumerate(e["probes"]):
            row = by_question[(e["episode_id"], i)]
            assert row["question"] == probe["question"] and row["gold"] == probe["gold_answer"]
            score = score_answer(row["predicted"], probe["gold_answer"], probe["probe_type"], spec=e["latent_spec"])
            assert score["match"] == row["match"] and score["reason"] == row["reason"]
            answers.append({"probe_type": probe["probe_type"], **score})
        official[e["episode_id"]] = {"answers": answers}
    assert Counter(r["match"] for r in rows) == {"correct": 191, "partial": 15, "incorrect": 18}
    assert metrics["summary"]["accuracy"] == 191 / 224
    for group_key, metric_key in (("probe_type", "by_type"), ("paradigm", "by_paradigm")):
        for group, expected in metrics[metric_key].items():
            subset = [r for r in rows if r[group_key] == group]
            counts = Counter(r["match"] for r in subset)
            assert len(subset) == expected["n"]
            for label in ("correct", "partial", "incorrect", "execution_error"):
                assert counts[label] == expected[label]
            assert expected["accuracy"] == counts["correct"] / len(subset)
    assert compute_baseline_metrics(official, build_episode_meta(episodes)) == metrics["native_metrics"]

    usage = metrics["answer_usage"]
    assert usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"] == 4931222
    for key, total in usage.items():
        assert sum(r["usage"][key] for r in rows) == total, key
    measures = {
        "per_question_input_tokens": [r["usage"]["input_tokens"] for r in rows],
        "per_question_output_tokens": [r["usage"]["output_tokens"] for r in rows],
        "per_question_total_tokens": [r["usage"]["total_tokens"] for r in rows],
        "per_question_first_call_input_tokens": [r["first_call_input_tokens"] for r in rows],
        "per_question_model_calls": [r["usage"]["requests"] for r in rows],
        "per_question_seconds": [r["seconds"] for r in rows],
        "per_question_read_calls": [r["tool_calls"].get("memory_read", 0) for r in rows],
        "per_question_steps": [r["steps"] for r in rows],
    }
    for key, values in measures.items():
        assert_stats(values, metrics["performance"][key])
    behavior = metrics["answer_behavior"]
    for tool, counts in behavior["tool_calls"].items():
        assert sum(r["tool_calls"].get(tool, 0) for r in rows) == counts["attempts"]
        assert sum(r["tool_calls"].get(tool, 0) > 0 for r in rows) == counts["task_count"]
    assert sum(sum(r["tool_calls"].values()) for r in rows) == behavior["total_tool_attempts"] == 1042
    assert_stats([r["initial_candidate_count"] for r in rows], behavior["initial_candidate_counts"])
    assert all(0 <= r["initial_candidate_count"] <= 5 for r in rows)
    assert_stats([r["cited_memory_count"] for r in rows], behavior["citation_counts"])
    for field, expected in (("runtime_verification", "runtime_verification_tasks"),
                            ("runtime_verification_tools", "runtime_verification_tool_tasks"),
                            ("runtime_verification_prompt_only", "runtime_verification_prompt_only_tasks"),
                            ("query_rewrite", "query_rewrite_tasks")):
        assert sum(bool(r[field]) for r in rows) == behavior[expected]

    snapshots = read(ROOT / "snapshots/manifest.json")
    assert set(snapshots["snapshots"]) == {e["episode_id"] for e in episodes}
    assert snapshots["answer_source_run"] == metrics["source_run"]
    assert snapshots["memory_source_run"] == metrics["reused_memory_construction"]["source_run"] == config["memory_source_run"]
    memories = []
    versions = 0
    for scene, entry in snapshots["snapshots"].items():
        path = ROOT / "snapshots" / entry["file"]
        assert sha(path) == entry["export_sha256"], scene
        snapshot = read(path)
        assert snapshot["episode_id"] == scene
        tables = snapshot["tables"]
        assert set(tables) == {"memories", "memory_versions"}
        assert tables["memories"]["columns"] == ["id", "data_json", "status", "updated_at"]
        assert tables["memory_versions"]["columns"] == ["id", "memory_id", "revision", "data_json", "reason", "created_at"]
        logical = [tables[t]["rows"] for t in ("memories", "memory_versions")]
        assert hashlib.sha256(json.dumps(logical, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == entry["memory_fingerprint"]
        assert len(logical[0]) == entry["memory_records"]
        assert len(logical[1]) == entry["version_records"]
        ids = {r[0] for r in logical[0]}
        assert len(ids) == len(logical[0])
        for row in rows:
            if row["episode_id"] == scene:
                assert set(row["initial_candidate_ids"]) <= ids
                assert (ROOT / "reports/memprobe" / row["snapshot"]).resolve() == path.resolve()
        memories.extend(json.loads(r[1]) for r in logical[0])
        versions += len(logical[1])
    construction = metrics["reused_memory_construction"]
    assert len(memories) == construction["records"] == 1603 and versions == 1982
    assert sum(s["history_user_turns"] for s in snapshots["snapshots"].values()) == construction["history_turns"]["sum"] == 2693
    for key in ("status", "fact_state", "decision_state", "acceptance_source", "conflict_status"):
        assert dict(Counter(m[key] for m in memories)) == construction[key], key
    for key, value in construction["ingestion_usage"].items():
        assert sum(v[key] for v in construction["ingestion_usage_by_phase"].values()) == value

    files = public_files()
    missing_links = []
    privacy_matches = []
    patterns = [r"\bsk-[A-Za-z0-9_-]{20,}", r"\b\d{5,}@(qq\.com|163\.com)\b",
                r"[A-Z]:[/\\]Personal agent", r"[A-Z]:[/\\]Users[/\\]Administrator",
                r"https://(?:shell\.)?wyzlab\.ai", r"\b(?:ou_|cli_)[A-Za-z0-9]{16,}"]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert path.suffix not in {".db", ".sqlite", ".sqlite3", ".pyc", ".log"}, relative
        assert path.name != ".env", relative
        text = path.read_text("utf-8-sig")
        assert "\ufffd" not in text, relative
        if any(re.search(p, text) for p in patterns):
            privacy_matches.append(relative)
        if path.suffix == ".py":
            ast.parse(text, filename=relative)
        if path.suffix == ".md":
            for match in re.finditer(r"\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)", text):
                url = match.group(1) or match.group(2)
                if re.match(r"^(https?://|mailto:|#)", url):
                    continue
                target = (path.parent / url.split("#")[0]).resolve()
                if not target.exists():
                    missing_links.append([relative, url])
    assert not privacy_matches, privacy_matches
    assert not missing_links, missing_links
    actual_paths = {p.relative_to(ROOT).as_posix() for p in files}
    expected_paths = expected_public_paths(episodes, manifest)
    assert actual_paths == expected_paths, {
        "missing": sorted(expected_paths - actual_paths),
        "unexpected": sorted(actual_paths - expected_paths),
    }
    provenance = config["publication_provenance"]
    assert provenance["selected_result"]["answer_source_run"] == metrics["source_run"]
    assert provenance["selected_result"]["memory_source_run"] == snapshots["memory_source_run"]
    assert provenance["publication_layout"]["main_documents"] == ["README.md", "TEST_REPORT.md"]
    assert set(p.relative_to(ROOT / "reports/memprobe").as_posix()
               for p in (ROOT / "reports/memprobe").rglob("*") if p.is_file()) == {
                   "metrics.json", "per_question.json", "configuration.json"}
    assert not (ROOT / "evals/memory").exists()
    assert not (ROOT / "evals/personal_agent_memory_eval_v1.json").exists()
    assert not (ROOT / "reports/regression69").exists()
    print(json.dumps({"memprobe_episodes": 56, "memprobe_questions": 224,
                      "selected_answers_rescored": 224, "correct": 191, "partial": 15, "incorrect": 18,
                      "native_metrics_recomputed": True, "per_question_statistics_recomputed": len(measures),
                      "logical_snapshots_verified": 56, "memory_records": len(memories), "memory_versions": versions,
                      "answer_tokens": usage["total_tokens"], "vendor_hashes_verified": len(manifest["files"]),
                      "public_files": len(files), "missing_links": missing_links,
                      "privacy_pattern_findings": privacy_matches, "real_model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Offline validation cannot access network")), \
         patch.object(socket.socket, "connect_ex", side_effect=RuntimeError("Offline validation cannot access network")):
        main()
