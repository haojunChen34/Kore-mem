"""Sequential histories and isolated, resumable native-memory evaluation."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta

from common import (HERE, RUN, VENDOR, REVISION, credentials, dump, digest, load_episodes,
                    network_setup, product_hashes, public_profile, redact, replay_units, sha256, utcnow, effective_models, validate_models)
from runtime import (BenchApp, CLOCK, DECAY_INTERVAL_TURNS, answer_question, ingest_unit, install_hooks,
                     memory_fingerprint, settings_for, snapshot_database)
from scoring import grade, schema
from timeout_policy import public_policy
from execution import read_plan, arms, counts, validate_baseline

FROZEN_FILES = ['common.py', 'execution.py', 'runtime.py', 'scoring.py', 'prepare.py', 'preflight.py', 'runner.py', 'aggregate_results.py', 'verify.py', 'timeout_policy.py', 'offline_checks.py', 'single_checks.py', 'readiness_checks.py', 'current_source_checks.py', 'check_memory_search.py', 'final_audit.py', 'reanswer.py', 'reanswer_checks.py']


def identity(saved, profiles):
    from kore_next.config import Settings
    settings = Settings()
    plan = read_plan()
    return {"dataset_revision": REVISION, "dataset_files": json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))["files"],
            "product_sources": product_hashes(), "adapter_sources": {p: sha256(HERE / p) for p in FROZEN_FILES},
            "profiles": {k: public_profile(v) for k, v in profiles.items()},
            "effective_models": effective_models(saved, profiles),
            "memory_policy": saved.get("_benchmark_memory_policy", {}),
            "roles": {k: saved.get(k) for k in ("default_provider", "reflection", "maintenance", "summary", "run_budget")},
            "scope": "native_post_run_long_term_memory_only",
            "evaluation": plan,
            "execution_subset": (json.loads((RUN / "execution_scope.json").read_text(encoding="utf-8"))
                                 if (RUN / "execution_scope.json").exists() else None),
            "requested_sampling": "Unchanged product requests; no explicit temperature/reasoning effort; effective provider defaults unverified",
            "timeout_policy": public_policy(), "synthetic_clock": f"2026-01-01T00:00:00Z + one minute per user turn; decay at the end of every {DECAY_INTERVAL_TURNS} turns",
            "answer_budget": {"source": "unchanged product saved budget and independent safety fuse",
                              "settings": {k: v for k, v in vars(settings).items() if k == "max_steps" or k.startswith("run_")},
                              "memory_tools": ["tool_catalog", "memory_list", "memory_search", "memory_read"]},
            "counterbalance": "not applicable: single native strategy" if plan["mode"] == "single" else "A/B order alternates by global question index", "retry": "Product transient request retry only; no answer-quality retries",
            "judge": "Unchanged pinned MemProbe rule scorer; all returned sub-scores retained",
            "deadline_hours": 23, "extra_report_reserve_hours": 1}


def validate_preflights(frozen, directory=RUN):
    for name in ("offline_preflight.json", "live_preflight.json"):
        path = directory / name
        if not path.exists():
            raise RuntimeError("Required preflight missing: " + name)
        check = json.loads(path.read_text(encoding="utf-8"))
        if check.get("status") != "passed" or check.get("configuration_fingerprint") != digest(frozen):
            raise RuntimeError("Preflight failed or stale: " + name)


def reviewed_post_run_failure(episode_id, unit_id, result):
    """Resume only an inspected, immutable failed turn; never repair or rerun it."""
    if (result.get("status") != "product_failure"
            or result.get("failure_scope") != "native_post_run"
            or result.get("memory_processing_succeeded") is not False):
        return False
    path = RUN / "reviewed_failures.json"
    if not path.exists():
        return False
    reviews = json.loads(path.read_text(encoding="utf-8")).get("failures", {})
    review = reviews.get(f"{episode_id}/{unit_id}", {})
    evidence = RUN / "episodes" / episode_id / "ingestion" / "turns" / unit_id / "result.json"
    return (review.get("action") == "retain_failure_and_continue"
            and bool(review.get("reason")) and bool(review.get("reviewed_at"))
            and evidence.exists() and review.get("result_sha256") == sha256(evidence)
            and json.loads(evidence.read_text(encoding="utf-8")) == result)


class Controller:
    def __init__(self, episodes, personal, saved, profiles, concurrency):
        self.episodes, self.personal, self.saved, self.profiles = episodes, personal, saved, profiles
        self.concurrency = concurrency
        self.selected_arms = arms()
        scope_path = RUN / "execution_scope.json"
        scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.exists() else {}
        self.pause_on_product_failure = scope.get("pause_on_product_failure", read_plan().get("pause_on_product_failure", False))
        self.stop = asyncio.Event()
        self.state_path = RUN / "progress.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "started_at": utcnow(), "status": "prepared", **counts(episodes, self.selected_arms),
            "arms": self.selected_arms, "episodes": {}, "blockers": []}
        if self.state.get("arms", ["A", "B"]) != self.selected_arms:
            raise RuntimeError("Cannot mix single and paired results in one run")
        self.started = datetime.fromisoformat(self.state["started_at"])

    def persist(self):
        self.state["updated_at"] = utcnow()
        dump(self.state_path, self.state)

    def block(self, reason, *, episode_id=None, kind="environment"):
        self.stop.set()
        item = {"reason": reason, "episode_id": episode_id, "kind": kind, "at": utcnow()}
        if not any(x["reason"] == reason and x["episode_id"] == episode_id for x in self.state["blockers"]):
            self.state["blockers"].append(item)
        self.state["status"] = "PAUSED"
        self.persist()

    def should_stop(self):
        if (RUN / "STOP_REQUESTED.json").exists():
            self.block("Explicit graceful stop requested", kind="control")
        if (datetime.now(UTC) - self.started).total_seconds() >= 23 * 3600:
            self.block("23-hour execution limit reached; preserve one hour for reporting", kind="duration")
        return self.stop.is_set()

    async def episode(self, e):
        eid = e["episode_id"]
        if self.should_stop():
            return
        state = self.state["episodes"].setdefault(eid, {"status": "queued", "ingested": 0, "answers": 0})
        if state["status"] == "complete":
            return
        directory = RUN / "episodes" / eid
        units, _ = replay_units(e)
        app = None
        begin = time.perf_counter()
        try:
            state.update(status="ingestion", user_turns=len(units), started_at=state.get("started_at", utcnow()))
            self.persist()
            app = BenchApp(settings_for(self.personal, directory / "ingestion"), self.saved, self.profiles)
            await app.initialize_benchmark()
            for index, unit in enumerate(units):
                if self.should_stop():
                    return
                result = await ingest_unit(app, unit, directory / "ingestion" / "turns" / unit["unit_id"], eid, self.profiles, index)
                recorded_failure = (result["status"] == "product_failure"
                                    and result.get("failure_scope") == "native_post_run"
                                    and result.get("memory_processing_succeeded") is False)
                if result["status"] != "completed" and not recorded_failure:
                    self.block(result.get("error", "History unit failed"), episode_id=eid)
                    return
                if recorded_failure:
                    failures = state.setdefault("post_run_failures", [])
                    if unit["unit_id"] not in failures:
                        failures.append(unit["unit_id"])
                state["ingested"] = index + 1
                self.persist()
                if (recorded_failure and self.pause_on_product_failure
                        and not reviewed_post_run_failure(eid, unit["unit_id"], result)):
                    self.block(result.get("error", "Memory processing failed; inspect before continuing"),
                               episode_id=eid, kind="product_failure_review")
                    return
            snapshot = directory / "snapshot.db"
            snap_meta_path = directory / "snapshot.json"
            if not snapshot.exists():
                snapshot_database(app.settings.database, snapshot)
                dump(snap_meta_path, {"sha256": sha256(snapshot), "memory_fingerprint": memory_fingerprint(snapshot),
                     "user_turns": len(units), "frozen_at": utcnow()})
                memories = [json.loads(x["data_json"]) for x in await app.db.fetchall("SELECT data_json FROM memories ORDER BY id")]
                dump(directory / "stored_memories.json", memories)
                dump(directory / "core_diagnostic_only.json", await app.memory_service.core_entries_detailed())
                dump(directory / "memory_audit.json", [dict(x) for x in await app.db.fetchall("SELECT * FROM audit_log ORDER BY created_at,id")])
                dump(directory / "memory_versions.json", [dict(x) for x in await app.db.fetchall("SELECT * FROM memory_versions ORDER BY id")])
            snap_meta = json.loads(snap_meta_path.read_text(encoding="utf-8"))
            if sha256(snapshot) != snap_meta["sha256"]:
                raise RuntimeError("Immutable episode snapshot changed")
            state["status"] = "answers"
            self.persist()
            ct = CLOCK.set(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=len(units)))
            try:
                for qi, probe in enumerate(e["probes"]):
                    if self.should_stop():
                        return
                    candidate_path = directory / "candidates" / f"q{qi}.json"
                    native = self.selected_arms == ["native"]
                    if not native and not candidate_path.exists():
                        candidates = await app.memory_service.relevant_items(probe["question"])
                        notices = await app.memory_service.context_notices(probe["question"])
                        dump(candidate_path, {"question": probe["question"], "candidates": candidates, "notices": notices,
                                              "fingerprint": digest([candidates, notices])})
                    if not native:
                        candidate_record = json.loads(candidate_path.read_text(encoding="utf-8"))
                        candidates = candidate_record["candidates"]
                        notices = candidate_record["notices"]
                        if digest([candidates, notices]) != candidate_record["fingerprint"]:
                            raise RuntimeError("Fixed candidate set changed")
                    global_index = next(i for i, ep in enumerate(self.episodes) if ep["episode_id"] == eid) * 4 + qi
                    for arm in arms(question_index=global_index):
                        if self.should_stop():
                            return
                        trial_dir = directory / "answers" / f"q{qi}" / arm
                        result_path = trial_dir / "trial/result.json"
                        grade_path = trial_dir / "score.json"
                        if result_path.exists():
                            outcome = json.loads(result_path.read_text(encoding="utf-8"))
                        else:
                            settings = settings_for(self.personal, trial_dir)
                            settings.ensure_dirs()
                            if settings.database.exists():
                                raise RuntimeError("Unreconciled answer database already exists")
                            snapshot_database(snapshot, settings.database)
                            trial = BenchApp(settings, self.saved, self.profiles, frozen=True, arm=arm, answer_schema=schema(probe["probe_type"]))
                            try:
                                await trial.initialize_benchmark()
                                if memory_fingerprint(settings.database) != snap_meta["memory_fingerprint"]:
                                    raise RuntimeError("Answer starting memory does not match frozen snapshot")
                                if not native:
                                    trial.fixed_candidates = copy.deepcopy(candidates)
                                    trial.fixed_notices = copy.deepcopy(notices)
                                outcome = await answer_question(trial, probe["question"], trial_dir / "trial", eid, qi, self.profiles)
                                outcome["starting_memory_fingerprint"] = snap_meta["memory_fingerprint"]
                                outcome["candidate_source"] = "product_runtime" if native else "fixed_shared_candidates"
                                dump(result_path, outcome)
                            finally:
                                await trial.mcp_manager.close()
                                trial.db._host_lock.release()
                        if native:
                            # Audit the real runtime output; never feed this file back as fixed input.
                            candidates, notices = outcome.get("candidates", []), outcome.get("memory_notices", [])
                            dump(candidate_path, {"question": probe["question"], "candidates": candidates,
                                 "notices": notices, "fingerprint": digest([candidates, notices]),
                                 "source": "product_runtime"})
                        if outcome["status"] == "environment_error":
                            self.block(outcome.get("error", "Answer environment error"), episode_id=eid)
                            return
                        if not grade_path.exists():
                            answer = outcome.get("answer", "")
                            sc = grade(answer if outcome["status"] == "completed" else "[ERROR: product execution failed]", probe, e["latent_spec"])
                            dump(grade_path, {"episode_id": eid, "probe_index": qi, "arm": arm,
                                "probe_type": probe["probe_type"], "question": probe["question"], "gold": probe["gold_answer"],
                                "predicted": answer, **sc, "execution_status": outcome["status"]})
                        state["answers"] = len(list((directory / "answers").glob("q*/*/score.json")))
                        self.persist()
                        if outcome["status"] == "product_failure" and self.pause_on_product_failure:
                            self.block(outcome.get("error", "Answer execution failed; inspect before continuing"),
                                       episode_id=eid, kind="product_failure_review")
                            return
                        if sha256(snapshot) != snap_meta["sha256"]:
                            raise RuntimeError("Frozen episode snapshot mutated during answering")
            finally:
                CLOCK.reset(ct)
            state.update(status="complete", completed_at=utcnow(), active_seconds=time.perf_counter() - begin)
            self.persist()
            print(json.dumps({"episode_complete": eid, "answers": state["answers"], "seconds": state["active_seconds"]}), flush=True)
        except Exception as exc:
            self.block(redact(exc, self.profiles), episode_id=eid)
        finally:
            if app is not None:
                await app.mcp_manager.close()
                app.db._host_lock.release()

    async def batch(self, episodes):
        queue = asyncio.Queue()
        for e in episodes:
            queue.put_nowait(e)
        async def worker():
            while not self.should_stop():
                try:
                    e = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self.episode(e)
                queue.task_done()
        await asyncio.gather(*(worker() for _ in range(self.concurrency)))

    def estimate(self, pilot_ids):
        audit = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))
        unit_results = [json.loads(p.read_text(encoding="utf-8")) for eid in pilot_ids
                        for p in (RUN / "episodes" / eid / "ingestion/turns").glob("*/result.json")]
        answer_results = [json.loads(p.read_text(encoding="utf-8")) for eid in pilot_ids
                          for p in (RUN / "episodes" / eid / "answers").glob("q*/*/trial/result.json")]
        total_seconds = sum(r["seconds"] for r in unit_results + answer_results)
        memory_per_user = sum(r["seconds"] for r in unit_results) / max(1, len(unit_results))
        answer_per_trial = sum(r["seconds"] for r in answer_results) / max(1, len(answer_results))
        from execution import scoped_episodes
        planned_episodes = scoped_episodes(self.episodes)
        planned_answers = counts(planned_episodes, self.selected_arms)["answers_planned"]
        planned_turns = sum(len(replay_units(e)[0]) for e in planned_episodes)
        projected = (memory_per_user * planned_turns + answer_per_trial * planned_answers) / self.concurrency
        calls = [c for r in unit_results + answer_results for c in r.get("calls", [])]
        usage = {k: sum((c.get("usage") or {}).get(k, 0) for c in calls) for k in ("input_tokens", "output_tokens", "total_tokens")}
        estimate = {"pilot_episodes": len(pilot_ids), "pilot_user_turns": len(unit_results),
                    "pilot_answers": len(answer_results), "observed_summed_seconds": total_seconds,
                    "formal_answers_planned": planned_answers,
                    "concurrency": self.concurrency, "memory_seconds_per_user_turn": memory_per_user,
                    "answer_seconds_per_trial": answer_per_trial, "pilot_usage": usage,
                    "projected_execution_hours": projected / 3600,
                    "conservative_hours_with_30pct_margin_and_report": projected / 3600 * 1.3 + 1,
                    "warning": "Pilot is small; larger/noisier episodes can take longer. Real progress and deadline monitored."}
        dump(RUN / "pilot_estimate.json", estimate)
        return estimate


async def main(args):
    if not args.allow_live:
        raise RuntimeError("Formal execution needs --allow-live and current live preflight; preparation is offline")
    os.environ["KORE_MEMPROBE_ALLOW_LIVE"] = "1"
    network_setup()
    install_hooks()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    personal, saved, profiles = credentials()
    validate_models(saved, profiles)
    frozen = identity(saved, profiles)
    validate_baseline(frozen["evaluation"])
    validate_preflights(frozen)
    baseline = json.loads((RUN / "product_baseline.json").read_text(encoding="utf-8"))
    if baseline["product_sources"] != frozen["product_sources"]:
        raise RuntimeError("Product code changed since baseline")
    formal_path = RUN / "formal_config.json"
    if formal_path.exists():
        if json.loads(formal_path.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Formal configuration changed; do not silently resume")
    else:
        dump(formal_path, frozen)
    es = load_episodes()
    controller = Controller(es, personal, saved, profiles, args.concurrency)
    if controller.state.get("blockers"):
        raise RuntimeError("Unresolved pause; reconcile blocker before resume")
    pilot_ids = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))["pilot_episode_ids"]
    if frozen["execution_subset"]:
        if not args.pilot_only or frozen["execution_subset"]["episode_ids"] != pilot_ids:
            raise RuntimeError("Frozen scope requires the exact pilot scenes and --pilot-only")
        controller.state.update(**counts([e for e in es if e["episode_id"] in pilot_ids], controller.selected_arms),
                                scope="fixed_pilot_regression")
    controller.state["status"] = "RUNNING_PILOT"
    controller.persist()
    await controller.batch([e for e in es if e["episode_id"] in pilot_ids])
    if not controller.stop.is_set():
        estimate = controller.estimate(pilot_ids)
        if args.pilot_only:
            controller.state["status"] = "PILOT_COMPLETE"
        elif estimate["conservative_hours_with_30pct_margin_and_report"] > 24:
            controller.block("Pilot projects over one day: " + str(round(estimate["conservative_hours_with_30pct_margin_and_report"], 2)) + " hours", kind="duration")
        else:
            controller.state["status"] = "RUNNING_FULL"
            controller.persist()
            await controller.batch([e for e in es if e["episode_id"] not in pilot_ids])
            if not controller.stop.is_set():
                controller.state["status"] = "COMPLETE"
    controller.state["product_sources_unchanged"] = product_hashes() == frozen["product_sources"]
    controller.persist()
    print(json.dumps(controller.state, ensure_ascii=False), flush=True)
    return 0 if controller.state["status"] in ("COMPLETE", "PILOT_COMPLETE") else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--allow-live", action="store_true")
    raise SystemExit(asyncio.run(main(p.parse_args())))
