"""Synthetic preflight. Default is fully offline; --live explicitly permits API use."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import uuid
from datetime import UTC, datetime

from common import (RUN, credentials, digest, dump, network_setup, product_hashes,
                    redact, utcnow, validate_models)
from runtime import (BenchApp, CLOCK, answer_question, ingest_unit, install_hooks,
                     memory_fingerprint, settings_for, snapshot_database)
from scoring import grade, schema
from execution import arms


async def live_checks(personal, saved, profiles, directory):
    """Synthetic native pipeline, never official benchmark answers or fixtures."""
    apps, checks = [], []
    token = CLOCK.set(datetime(2026, 1, 1, tzinfo=UTC))
    def check(name, value, detail=None):
        checks.append({"name": name, "passed": bool(value), "detail": detail})
        if not value:
            raise RuntimeError(name + ": " + str(detail))
    try:
        app = BenchApp(settings_for(personal, directory / "ingestion"), saved, profiles)
        apps.append(app)
        await app.initialize_benchmark()
        for i, value in enumerate(("Zephyr", "Quartz")):
            text = f"Please remember my approved permanent architecture decision: Project Alder uses {value} as its database engine."
            if i:
                text += " This is an explicit update replacing the former Zephyr decision."
            r = await ingest_unit(app, {"unit_id": f"u{i}", "user": text,
                "assistant": "Understood. I recorded your confirmed architecture decision."},
                directory / "ingestion" / f"u{i}", "synthetic-preflight", profiles, i)
            check(f"live_ingestion_{i}", r["status"] == "completed", r.get("error"))
        candidates = await app.memory_service.relevant_items("What database does Project Alder use?")
        notices = await app.memory_service.context_notices("What database does Project Alder use?")
        check("native_top_five_limit", len(candidates) <= 5)
        snapshot = directory / "snapshot.db"
        snapshot_database(app.settings.database, snapshot)
        fingerprint = memory_fingerprint(snapshot)
        probe = {"question": "What database does Project Alder use?", "probe_type": "current_value", "gold_answer": "Quartz"}
        for arm in arms():
            settings = settings_for(personal, directory / arm)
            settings.ensure_dirs()
            snapshot_database(snapshot, settings.database)
            trial = BenchApp(settings, saved, profiles, frozen=True, arm=arm, answer_schema=schema("current_value"))
            apps.append(trial)
            await trial.initialize_benchmark()
            if arm != "native":
                trial.fixed_candidates, trial.fixed_notices = copy.deepcopy(candidates), copy.deepcopy(notices)
            check(arm + "_same_start", memory_fingerprint(settings.database) == fingerprint)
            result = await answer_question(trial, probe["question"], directory / arm / "trial", "synthetic-preflight", 0, profiles)
            check(arm + "_transport_and_runtime_complete", result["status"] == "completed", result.get("error"))
            # Correctness is diagnostic, not a permission to retry until a good answer.
            dump(directory / arm / "synthetic_score.json", grade(result.get("answer", ""), probe, {}))
            paths = (directory / arm / "trial/model_calls").glob("*.json")
            calls = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            check(arm + "_real_usage_present", bool(calls) and all(c["status"] != "completed" or c.get("usage", {}).get("input_tokens", 0) > 0 for c in calls))
        check("frozen_snapshot_unchanged", memory_fingerprint(snapshot) == fingerprint)
        return checks
    finally:
        for app in reversed(apps):
            await app.mcp_manager.close()
            app.db.release_lock()
        CLOCK.reset(token)


async def main(live=False):
    install_hooks()
    if live:
        os.environ["KORE_MEMPROBE_ALLOW_LIVE"] = "1"
    personal, saved, profiles = credentials()
    models = validate_models(saved, profiles)
    from runner import identity
    frozen = identity(saved, profiles)
    fingerprint = digest(frozen)
    report_path = RUN / ("live_preflight.json" if live else "offline_preflight.json")
    report = {"started_at": utcnow(), "status": "running", "live": live,
              "configuration_fingerprint": fingerprint, "configuration": frozen,
              "models": models, "checks": [], "official_answers_executed": 0}
    before = product_hashes()
    try:
        if live:
            offline = json.loads((RUN / "offline_preflight.json").read_text(encoding="utf-8"))
            if offline.get("status") != "passed" or offline.get("configuration_fingerprint") != fingerprint:
                raise RuntimeError("Run the current offline preflight first")
            os.environ["KORE_MEMPROBE_ALLOW_LIVE"] = "1"
            network_setup()
            directory = RUN / "synthetic" / ("live-" + uuid.uuid4().hex)
            report["directory"] = str(directory)
            report["checks"] = await live_checks(personal, saved, profiles, directory)
        else:
            from offline_checks import run_checks
            await run_checks(report["checks"])
            from single_checks import run_checks as single_checks
            await single_checks(report["checks"])
            from readiness_checks import run_checks as readiness_checks
            await readiness_checks(report["checks"])
            from current_source_checks import run_checks as current_source_checks
            await current_source_checks(report["checks"])
            report.update(real_api_calls=0, real_api_tokens=0,
                          mock_usage="Synthetic fixtures only; excluded from benchmark usage and scores")
        report["checks"].append({"name": "product_sources_unchanged", "passed": product_hashes() == before})
        report["checks"].append({"name": "configuration_unchanged_during_check", "passed": digest(identity(saved, profiles)) == fingerprint})
        report["status"] = "passed" if all(c["passed"] for c in report["checks"]) else "failed"
    except Exception as exc:
        report.update(status="failed", blocker={"type": type(exc).__name__, "message": redact(exc, profiles)})
    report["finished_at"] = utcnow()
    # Preserve every preflight attempt; the latest pointer never grants stale approval.
    dump(RUN / "preflight_attempts" / (("live-" if live else "offline-") + uuid.uuid4().hex + ".json"), report)
    dump(report_path, report)
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "blocker": report.get("blocker"),
                      "live": live, "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args().live)))
