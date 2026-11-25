"""Offline checks of public JSON restoration and new answers through real memory tools."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import httpx

from common import CONFIG, credentials, dump, load_episodes, offline_network_guard, product_hashes, sha256
from runtime import BenchApp, CLOCK, OFFLINE_MOCK, install_hooks, memory_fingerprint, settings_for
from timeout_policy import ObservedHTTPClient
from reanswer import (SNAPSHOTS, checked_snapshot, close, frozen_identity, question_clock, read, restore,
                      run_answers, score_result, select_tasks, summarize, trial_path)


async def run_checks(checks=None):
    checks = [] if checks is None else checks
    def check(name, value):
        checks.append({"name": "snapshot_answers_" + name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)
    def rejects(name, action):
        try:
            action()
        except (ValueError, KeyError):
            check(name, True)
        else:
            check(name, False)
    install_hooks()
    before = {p.name: sha256(p) for p in SNAPSHOTS.glob("*.json")}
    product_before = product_hashes()
    episodes = load_episodes()
    personal, saved, profiles = credentials()
    mock = OFFLINE_MOCK.set(True)
    try:
        with offline_network_guard() as network, tempfile.TemporaryDirectory(prefix="kore-snapshot-answers-") as tmp:
            root = Path(tmp)
            for episode in episodes:
                eid = episode["episode_id"]
                data, meta = checked_snapshot(eid)
                token = CLOCK.set(question_clock(meta))
                app = BenchApp(settings_for(personal, root / "restore" / eid), saved, profiles, frozen=True)
                try:
                    await app.initialize_benchmark()
                    await restore(app, data, meta)
                    check(eid + "_exact_rows", memory_fingerprint(app.settings.database) == meta["memory_fingerprint"])
                    counts = [len(await app.db.fetchall("SELECT memory_id FROM " + table)) for table in ("memories_fts", "memory_source_fts")]
                    check(eid + "_both_indexes", counts == [meta["memory_records"]] * 2)
                    candidates = await app.memory_service.relevant_items(episode["probes"][0]["question"])
                    check(eid + "_native_search", len(candidates) <= 5 and all(m["id"] in {r[0] for r in data["tables"]["memories"]["rows"]} for m in candidates))
                    check(eid + "_isolated", not await app.db.fetchall("SELECT id FROM messages")
                          and not (await app.db.get_json("privacy_settings"))["auto_memory_enabled"])
                finally:
                    await close(app)
                    CLOCK.reset(token)
            data, meta = checked_snapshot(episodes[0]["episode_id"])
            app = BenchApp(settings_for(personal, root / "rollback"), saved, profiles, frozen=True)
            try:
                await app.initialize_benchmark()
                with patch.object(app.memory_service, "_index", side_effect=RuntimeError("index failed")):
                    try:
                        await restore(app, data, meta)
                    except RuntimeError:
                        pass
                    else:
                        check("rollback_error_expected", False)
                check("atomic_rollback", not await app.db.fetchall("SELECT id FROM memories")
                      and not await app.db.fetchall("SELECT id FROM memory_versions"))
                await restore(app, data, meta)
                try:
                    await restore(app, data, meta)
                except ValueError:
                    check("nonempty_restore_rejected", True)
                else:
                    check("nonempty_restore_rejected", False)
            finally:
                await close(app)

            # Corruption checks use temporary copies, never modify public evidence.
            bad_root = root / "bad"
            manifest = read(SNAPSHOTS / "manifest.json")
            eid = episodes[0]["episode_id"]
            dump(bad_root / "manifest.json", manifest)
            dump(bad_root / (eid + ".json"), {})
            rejects("bad_hash", lambda: checked_snapshot(eid, bad_root))
            malformed = copy.deepcopy(data)
            malformed["tables"]["memories"]["columns"][0] = "unexpected"
            dump(bad_root / (eid + ".json"), malformed)
            manifest["snapshots"][eid]["export_sha256"] = sha256(bad_root / (eid + ".json"))
            dump(bad_root / "manifest.json", manifest)
            rejects("bad_schema", lambda: checked_snapshot(eid, bad_root))
            manifest["snapshots"][eid]["file"] = "../elsewhere.json"
            dump(bad_root / "manifest.json", manifest)
            rejects("path_escape", lambda: checked_snapshot(eid, bad_root))
            rejects("unknown_scene", lambda: select_tasks(episodes, ["NOT-A-SCENE"]))
            rejects("invalid_probe", lambda: select_tasks(episodes, [eid], 99))
            rejects("ambiguous_probe", lambda: select_tasks(episodes, [e["episode_id"] for e in episodes[:2]], 0))
            check("all_scope", len(select_tasks(episodes)) == 224)

            # This model fixture knows no answers: it searches/reads actual imported memory,
            # then emits a unique marker, proving published answers are never reused.
            selected = copy.deepcopy(episodes[0])
            for probe in selected["probes"]:
                probe["gold_answer"] = "GOLD_MUST_NOT_REACH_MODEL_184a"
            tasks = select_tasks([selected])
            wire = []
            fault = {"missing_usage": False}
            def handler(request):
                body = json.loads(request.content)
                wire.append(body)
                messages = body["messages"]
                tools = [m for m in messages if m.get("role") == "tool"]
                names = [m.get("name") for m in tools]
                content, calls = '{"answer":"offline_new_answer_184a"}', []
                rewrite = messages[0]["content"].startswith("Rewrite only the requested property")
                if rewrite:
                    content = '{"query":"stored details"}'
                elif "tool_catalog" not in names:
                    calls = [("tool_catalog", {"query": "memory"})]
                elif "memory_search" not in names:
                    title = json.loads(data["tables"]["memories"]["rows"][0][1])["title"]
                    calls = [("memory_search", {"query": title, "purpose": "all", "limit": 5})]
                elif "memory_read" not in names:
                    calls = [("memory_read", {"memory_id": data["tables"]["memories"]["rows"][0][0], "purpose": "all"})]
                message = {"role": "assistant", "content": "" if calls else content}
                if calls:
                    message["tool_calls"] = [{"id": "offline-" + str(len(wire)), "type": "function", "function":
                                             {"name": name, "arguments": json.dumps(args)}} for name, args in calls]
                usage = {} if fault["missing_usage"] else {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
                if body.get("stream"):
                    delta = {"content": message["content"]}
                    if calls:
                        delta["tool_calls"] = [{"index": i, **t} for i, t in enumerate(message["tool_calls"])]
                    return httpx.Response(200, text="data: " + json.dumps({"choices": [{"index": 0, "delta": delta}], "usage": usage}) + "\n\ndata: [DONE]\n\n",
                                          headers={"content-type": "text/event-stream"})
                return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})
            def client(*args, **kwargs):
                return ObservedHTTPClient(*args, **kwargs, transport=httpx.MockTransport(handler))
            run = root / "new-answers"
            with patch.dict(os.environ, {CONFIG.get("api_key_env", "KORE_MEM_API_KEY"): "offline-mock-fixture",
                                         "KORE_MEMPROBE_ALLOW_LIVE": "0"}), patch("reanswer.install_hooks"), patch.object(httpx, "AsyncClient", client):
                # Hooks are installed above; avoid replacing this MockTransport when
                # the real public runner performs its ordinary hook initialization.
                metrics = await run_answers(run, tasks)
                check("new_answers_complete", metrics["complete"] and metrics["summary"]["planned"] == 4)
                count = len(wire)
                resumed = await run_answers(run, tasks, resume=True)
                check("resume_no_repeat", len(wire) == count and resumed["summary"] == metrics["summary"])
                failure_run = root / "missing-usage"
                fault["missing_usage"] = True
                failed = await run_answers(failure_run, tasks)
                check("missing_usage_pauses", not failed["complete"] and not failed["all_questions_recorded"]
                      and failed["summary"]["planned"] == 4
                      and failed["summary"]["labels"] == {"execution_error": 1, "not_scored": 3})
                failed_result = trial_path(failure_run, eid, 0) / "trial/result.json"
                failed_hash = sha256(failed_result)
                fault["missing_usage"] = False
                continued = await run_answers(failure_run, tasks, resume=True)
                check("resume_keeps_failure", sha256(failed_result) == failed_hash and continued["all_questions_recorded"]
                      and not continued["complete"] and continued["summary"]["labels"]["execution_error"] == 1)
                interrupted = root / "interrupted"
                dump(interrupted / "reanswer_config.json", frozen_identity(tasks))
                dump(trial_path(interrupted, eid, 0) / "trial/inflight.json", {"started": True})
                count = len(wire)
                try:
                    await run_answers(interrupted, tasks, resume=True)
                except RuntimeError as exc:
                    check("interrupted_no_resend", "Unreconciled" in str(exc) and len(wire) == count)
                else:
                    check("interrupted_no_resend", False)
            results = [read(trial_path(run, eid, qi) / "trial/result.json") for _, qi in tasks]
            check("fresh_answers", all("offline_new_answer_184a" in r["answer"] for r in results))
            check("actual_tools", all({"memory_search", "memory_read"} <= {t["name"] for t in r["tool_observations"]} for r in results))
            check("gold_not_in_requests", "GOLD_MUST_NOT_REACH_MODEL_184a" not in json.dumps(wire))
            check("clock_restored", all(r["question_clock"] == question_clock(meta).isoformat() for r in results))
            check("independent_start", all(r["starting_memory_fingerprint"] == meta["memory_fingerprint"] for r in results))
            check("real_cost_zero", metrics["answer_usage"]["requests"] == 0 and metrics["memory_construction"]["new_tokens"] == 0)
            check("no_ingestion", not list(run.glob("**/ingestion")))
            check("json_only", not list(run.glob("**/*.md")))
            try:
                await run_answers(run, tasks)
            except FileExistsError:
                check("existing_run_rejected", True)
            else:
                check("existing_run_rejected", False)
            try:
                await run_answers(run, tasks[:1], resume=True)
            except ValueError:
                check("changed_scope_rejected", True)
            else:
                check("changed_scope_rejected", False)
            check("errors_not_correct", score_result({"status": "environment_error", "answer": "GOLD_MUST_NOT_REACH_MODEL_184a"}, selected, 0)["match"] == "execution_error")
            check("empty_not_correct", score_result({"status": "completed", "answer": ""}, selected, 0)["match"] == "execution_error")
            check("network_blocked", not network)
    finally:
        OFFLINE_MOCK.reset(mock)
    check("source_snapshots_unchanged", before == {p.name: sha256(p) for p in SNAPSHOTS.glob("*.json")})
    check("product_sources_unchanged", product_before == product_hashes())
    return checks


if __name__ == "__main__":
    results = asyncio.run(run_checks())
    print(json.dumps({"status": "passed", "checks": len(results), "real_api_calls": 0, "real_api_tokens": 0}))
