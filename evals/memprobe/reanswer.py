"""Answer from published logical snapshots; never ingest history or overwrite evidence."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path

from common import (CONFIG, HERE, PROJECT_ROOT, RUN, VENDOR, credentials, digest, dump,
                    load_episodes, network_setup, product_hashes, redact, sha256,
                    utcnow, validate_models)
from runtime import (BenchApp, CLOCK, OFFLINE_MOCK, ReadOnlyMemoryTools, answer_question,
                     install_hooks, memory_fingerprint, settings_for, snapshot_database)
from scoring import build_episode_meta, compute_baseline_metrics, grade, schema
from aggregate_results import observed_usage, percentile
from verify import valid_answer_input

SNAPSHOTS = PROJECT_ROOT / "snapshots"
FORMAT = "kore-mem-logical-snapshot-v1"
TABLES = {
    "memories": ["id", "data_json", "status", "updated_at"],
    "memory_versions": ["id", "memory_id", "revision", "data_json", "reason", "created_at"],
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_snapshot(episode_id, root=SNAPSHOTS):
    """Validate before touching a database; column names are a closed allowlist."""
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported snapshot manifest format")
    meta = manifest["snapshots"][episode_id]
    path = (root / meta["file"]).resolve()
    if path.parent != root or path.name != episode_id + ".json":
        raise ValueError("Snapshot path must name one scene inside the snapshot directory")
    if sha256(path) != meta["export_sha256"]:
        raise ValueError("Snapshot export hash mismatch: " + episode_id)
    data = read(path)
    if data.get("format") != FORMAT or data.get("episode_id") != episode_id or set(data["tables"]) != set(TABLES):
        raise ValueError("Snapshot identity/tables mismatch")
    all_rows = []
    for name, columns in TABLES.items():
        table = data["tables"][name]
        if table["columns"] != columns:
            raise ValueError("Unsupported snapshot columns: " + name)
        rows = table["rows"]
        if any(not isinstance(r, list) or len(r) != len(columns) for r in rows):
            raise ValueError("Malformed snapshot rows")
        ids = [r[0] for r in rows]
        if any(not isinstance(i, str) for i in ids) or ids != sorted(set(ids)):
            raise ValueError("Snapshot IDs must be unique and sorted")
        for row in rows:
            record = json.loads(row[columns.index("data_json")])
            if not isinstance(record, dict) or (name == "memories" and record.get("id") != row[0]):
                raise ValueError("Invalid memory JSON/identity")
        all_rows.append(rows)
    if digest(all_rows) != meta["memory_fingerprint"]:
        raise ValueError("Snapshot logical fingerprint mismatch")
    if [len(r) for r in all_rows] != [meta["memory_records"], meta["version_records"]]:
        raise ValueError("Snapshot record counts mismatch")
    if any(r[1] not in {m[0] for m in all_rows[0]} for r in all_rows[1]):
        raise ValueError("Version refers to an absent memory")
    if type(meta["history_user_turns"]) is not int or meta["history_user_turns"] < 0:
        raise ValueError("Invalid snapshot clock")
    return data, meta


def question_clock(meta):
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=meta["history_user_turns"])


async def close(app):
    try:
        await app.mcp_manager.close()
    finally:
        app.db.release_lock()


async def restore(app, data, meta):
    """Restore exact rows atomically, rebuilding only derived indexes in product code."""
    if not app.frozen:
        raise ValueError("Snapshot answering requires frozen/no-learning mode")
    db = app.memory_service.db
    async with db.atomic():
        for name, columns in TABLES.items():
            actual = [r["name"] for r in await db.fetchall("PRAGMA table_info(" + name + ")")]
            if actual != columns or await db.fetchall("SELECT id FROM " + name + " LIMIT 1"):
                raise ValueError("Restore requires compatible, empty memory tables")
        for name, columns in TABLES.items():
            sql = "INSERT INTO " + name + "(" + ",".join(columns) + ") VALUES(" + ",".join("?" for _ in columns) + ")"
            for row in data["tables"][name]["rows"]:
                await db.execute(sql, tuple(row))
        await db.execute("DELETE FROM memories_fts")
        await db.set_json("memory_evidence_index_version", 0)
        await app.memory_service._ensure_evidence_index()
        rows = [list(map(list, await db.fetchall("SELECT * FROM " + name + " ORDER BY id"))) for name in TABLES]
        if digest(rows) != meta["memory_fingerprint"]:
            raise RuntimeError("Restore changed memory contents")
    if memory_fingerprint(app.settings.database) != meta["memory_fingerprint"]:
        raise RuntimeError("Committed memory fingerprint mismatch")


def select_tasks(episodes, scene_ids=None, probe_index=None):
    by_id = {e["episode_id"]: e for e in episodes}
    ids = list(by_id) if scene_ids is None else scene_ids
    if not ids or len(set(ids)) != len(ids) or any(eid not in by_id for eid in ids):
        raise ValueError("Supply unique existing episode IDs")
    if probe_index is not None and (len(ids) != 1 or not 0 <= probe_index < len(by_id[ids[0]]["probes"])):
        raise ValueError("--probe-index requires one episode and a valid zero-based index")
    return [(by_id[eid], qi) for eid in ids for qi in range(len(by_id[eid]["probes"]))
            if probe_index is None or qi == probe_index]


def frozen_identity(tasks):
    from runner import FROZEN_FILES
    return {"mode": "published_snapshot_answers", "history_reingestion": False, "answer_learning": False,
            "mock": OFFLINE_MOCK.get(), "configuration": CONFIG,
            "product_sources": product_hashes(), "adapter_sources": {f: sha256(HERE / f) for f in FROZEN_FILES},
            "vendor_manifest_sha256": sha256(VENDOR / "manifest.json"),
            "snapshot_manifest_sha256": sha256(SNAPSHOTS / "manifest.json"),
            "source_exports": {e["episode_id"]: checked_snapshot(e["episode_id"])[1] for e, _ in tasks},
            "questions": [[e["episode_id"], qi] for e, qi in tasks],
            "task_fingerprint": digest([[e["episode_id"], qi, e["probes"][qi], e["latent_spec"]] for e, qi in tasks]),
            "scope": "Native retrieval and read tools over published memories; not extraction evaluation",
            "retry": "Product transport retries only; a recorded answer is never re-executed",
            "clock": "2026-01-01T00:00:00Z + snapshot history_user_turns minutes",
            "cost_scope": "New answer calls only; historical memory construction is not rerun"}


def trial_path(directory, eid, qi):
    return directory / "episodes" / eid / "answers" / f"q{qi}" / "native"


def score_result(result, episode, qi):
    probe = episode["probes"][qi]
    score = (grade(result.get("answer", ""), probe, episode["latent_spec"])
             if result.get("status") == "completed" else {"match": "execution_error", "reason": "Execution did not complete"})
    return {"episode_id": episode["episode_id"], "probe_index": qi, "probe_type": probe["probe_type"],
            "question": probe["question"], "gold": probe["gold_answer"], "predicted": result.get("answer"), **score}


def summarize(directory, tasks, frozen):
    """Rescore saved outputs, audit inputs and account for even interrupted calls. JSON only."""
    if ([[e["episode_id"], qi] for e, qi in tasks] != frozen["questions"] or
            digest([[e["episode_id"], qi, e["probes"][qi], e["latent_spec"]] for e, qi in tasks]) != frozen["task_fingerprint"]):
        raise ValueError("Summary requires the original question selection and scoring inputs")
    rows, calls, checks, official = [], [], [], {}
    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
    check("configuration_and_sources_unchanged", frozen_identity(tasks) == frozen)
    for episode, qi in tasks:
        eid, probe = episode["episode_id"], episode["probes"][qi]
        base = trial_path(directory, eid, qi)
        result = read(base / "trial/result.json") if (base / "trial/result.json").exists() else {}
        ledger = [read(p) for p in sorted((base / "trial/model_calls").glob("*.json"))]
        calls.extend(ledger)
        meta = frozen["source_exports"][eid]
        if result:
            score = score_result(result, episode, qi)
            if (base / "score.json").exists():
                check(eid + f"/q{qi}/score", read(base / "score.json") == score)
            else:
                dump(base / "score.json", score)
            official.setdefault(eid, {"answers": []})["answers"].append(score)
            check(eid + f"/q{qi}/start", result.get("starting_memory_fingerprint") == meta["memory_fingerprint"]
                  and result.get("question_clock") == question_clock(meta).isoformat()
                  and result.get("candidate_source") == "product_runtime")
            check(eid + f"/q{qi}/ledger", {c["id"] for c in ledger} == {c["id"] for c in result.get("calls", [])})
        else:
            score = {"match": "not_scored"}
        for call in ledger:
            # Native context summaries are logged but use their own prompt format.
            check(call["id"] + "/scope", call.get("phase") in {"answer", "context_summary"}
                  and bool(call.get("offline_mock_transport")) == frozen["mock"]
                  and (call.get("phase") == "context_summary" or valid_answer_input(call, probe["question"], result))
                  and all(t["function"]["name"] in ReadOnlyMemoryTools.ALLOWED for t in call["request"]["tools"]))
        tools = result.get("tool_observations", [])
        check(eid + f"/q{qi}/tools", all(t["name"] in ReadOnlyMemoryTools.ALLOWED for t in tools))
        rows.append({"episode_id": eid, "probe_index": qi, "probe_type": probe["probe_type"],
                     "execution_status": result.get("status", "not_executed"), "score": score, "match": score["match"],
                     "correct": result.get("status") == "completed" and score["match"] == "correct",
                     "answer": result.get("answer"), "question": probe["question"],
                     "usage": observed_usage(ledger), "seconds": result.get("seconds"), "tools": tools,
                     "snapshot": f"snapshots/{eid}.json", "result": (base / "trial/result.json").relative_to(directory).as_posix()})
    for eid, meta in frozen["source_exports"].items():
        path = directory / "episodes" / eid / "snapshot.db"
        if path.exists():
            check(eid + "/frozen_memory", memory_fingerprint(path) == meta["memory_fingerprint"])
    check("unique_model_call_ids", len(calls) == len({c["id"] for c in calls}))
    def distribution(items):
        return {"planned": len(items), "correct": sum(r["correct"] for r in items),
                "labels": dict(Counter(r["match"] for r in items)),
                "accuracy": sum(r["correct"] for r in items) / len(items) if items else None}
    times = [r["seconds"] for r in rows if r["seconds"] is not None]
    episodes = list({e["episode_id"]: e for e, _ in tasks}.values())
    audit = {"status": "passed" if all(c["passed"] for c in checks) else "failed", "checks": checks}
    metrics = {"mode": frozen["mode"], "mock": frozen["mock"], "summary": distribution(rows),
               "all_questions_recorded": all(r["execution_status"] != "not_executed" for r in rows),
               "complete": all(r["execution_status"] == "completed" for r in rows) and audit["status"] == "passed",
               "by_type": {t: distribution([r for r in rows if r["probe_type"] == t]) for t in sorted({r["probe_type"] for r in rows})},
               "by_episode": {e["episode_id"]: distribution([r for r in rows if r["episode_id"] == e["episode_id"]]) for e in episodes},
               "native": compute_baseline_metrics(official, build_episode_meta(episodes)),
               "native_denominator_note": "Upstream metrics exclude execution errors; summary accuracy includes all planned questions",
               "answer_usage": observed_usage(calls), "scoring_usage": {"model_calls": 0, "tokens": 0},
               "memory_construction": {"executed": False, "new_model_calls": 0, "new_tokens": 0},
               "seconds": {"mean": sum(times) / len(times) if times else None, "p50": percentile(times, .5), "p95": percentile(times, .95)},
               "tools": dict(Counter(t["name"] for r in rows for t in r["tools"])), "audit_status": audit["status"]}
    for name, value in {"metrics": metrics, "per_question": rows, "integrity_audit": audit,
                        "official_format_results": official, "model_call_index": [{k: c.get(k) for k in
                          ("id", "episode_id", "unit_id", "phase", "status", "usage", "seconds", "error")} for c in calls]}.items():
        dump(directory / "reports" / (name + ".json"), value)
    return metrics


async def run_answers(directory, tasks, *, resume=False):
    directory = Path(directory).resolve()
    # The public CLI uses runs/<name>; internal callers must also stay away from published evidence.
    for protected in (SNAPSHOTS, PROJECT_ROOT / "reports", HERE / "vendor"):
        if directory.is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(directory):
            raise ValueError("Run directory overlaps published evidence")
    personal, saved, profiles = credentials()
    validate_models(saved, profiles)
    frozen = frozen_identity(tasks)
    config_path = directory / "reanswer_config.json"
    if resume:
        if not config_path.exists() or read(config_path) != frozen:
            raise ValueError("Resume requires the same frozen selection, code, model and configuration")
        if summarize(directory, tasks, frozen)["audit_status"] != "passed":
            raise ValueError("Existing run failed its integrity audit; inspect before continuing")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        dump(config_path, frozen)
    install_hooks()
    network_setup()
    try:
        for episode, qi in tasks:
            eid, probe = episode["episode_id"], episode["probes"][qi]
            base = trial_path(directory, eid, qi)
            if (base / "trial/result.json").exists():
                continue  # Includes errors: no selective quality retries or hidden discarded attempts.
            if base.exists():
                raise RuntimeError("Unreconciled interrupted question; inspect its logs before starting a new run: " + str(base))
            if (directory / "STOP_REQUESTED.json").exists():
                break
            data, meta = checked_snapshot(eid)
            token = CLOCK.set(question_clock(meta))
            app = None
            try:
                snapshot = directory / "episodes" / eid / "snapshot.db"
                if not snapshot.exists():
                    seed_settings = settings_for(personal, directory / "episodes" / eid / "restore")
                    if seed_settings.database.exists():
                        raise RuntimeError("Unreconciled snapshot restoration; keep this run for inspection")
                    app = BenchApp(seed_settings, saved, profiles, frozen=True)
                    await app.initialize_benchmark()
                    await restore(app, data, meta)
                    snapshot_database(app.settings.database, snapshot)
                    await close(app)
                    app = None
                if memory_fingerprint(snapshot) != meta["memory_fingerprint"]:
                    raise RuntimeError("Restored scene snapshot mutated")
                settings = settings_for(personal, base)
                settings.ensure_dirs()
                snapshot_database(snapshot, settings.database)
                app = BenchApp(settings, saved, profiles, frozen=True, arm="native", answer_schema=schema(probe["probe_type"]))
                await app.initialize_benchmark()
                if memory_fingerprint(settings.database) != meta["memory_fingerprint"]:
                    raise RuntimeError("Independent question copy differs from frozen memory")
                # Controller holds the gold; only the question and format schema reach the runtime.
                result = await answer_question(app, probe["question"], base / "trial", eid, qi, profiles)
                result.update(starting_memory_fingerprint=meta["memory_fingerprint"], candidate_source="product_runtime",
                              question_clock=question_clock(meta).isoformat())
                dump(base / "trial/result.json", result)
                dump(base / "score.json", score_result(result, episode, qi))
                if result["status"] != "completed":
                    dump(directory / "pause.json", {"at": utcnow(), "episode_id": eid, "probe_index": qi,
                         "reason": result.get("error", result["status"]), "resume": "Keep this failure and continue remaining questions"})
                    break
            finally:
                if app is not None:
                    await close(app)
                CLOCK.reset(token)
    except BaseException as exc:
        dump(directory / "pause.json", {"at": utcnow(), "reason": redact(exc, profiles)})
        raise
    finally:
        summarize(directory, tasks, frozen)
    return read(directory / "reports/metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--episode", action="append", help="Scene ID; repeat to select several scenes")
    scope.add_argument("--all", action="store_true", help="All 56 scenes / 224 questions")
    parser.add_argument("--probe-index", type=int, help="One zero-based question index, with one --episode")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-live", action="store_true", help="Authorize paid answer API calls")
    mode.add_argument("--verify", action="store_true", help="Offline rescore/audit of this answer-only run")
    parser.add_argument("--resume", action="store_true", help="Continue only unattempted questions in an identical run")
    args = parser.parse_args()
    tasks = select_tasks(load_episodes(), args.episode, args.probe_index)
    if args.verify:
        metrics = summarize(RUN, tasks, read(RUN / "reanswer_config.json"))
    else:
        os.environ["KORE_MEMPROBE_ALLOW_LIVE"] = "1"
        metrics = asyncio.run(run_answers(RUN, tasks, resume=args.resume))
    print(json.dumps({"output": str(RUN), "complete": metrics["complete"], "summary": metrics["summary"],
                      "audit": metrics["audit_status"]}, ensure_ascii=False))
    return 0 if metrics["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
