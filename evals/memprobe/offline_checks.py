"""Adapter integration checks with real services and HTTP MockTransport only.

These controlled responses verify plumbing, not the model's semantic accuracy.
No official question, answer or personal memory is used in the fixtures.
"""
from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx

from common import INITIAL_RUN, dump, offline_network_guard, replay_units
from check_memory_search import close, isolated_settings, route_checks, seed
from kore_next import application as application_module
from kore_next import memory as memory_module
from kore_next import memory_protocol as protocol
from kore_next.memory import MemoryService
from kore_next.util import dumps
from runtime import (BenchApp, BenchMemory, CALL_CONTEXT, CLOCK, OFFLINE_MOCK,
                     ObservedModel, ReadOnlyMemoryTools, answer_question, call_context,
                     ingest_unit, install_hooks, memory_fingerprint, memory_now, replay_source_records,
                     snapshot_database)
from scoring import grade, schema
from timeout_policy import ObservedHTTPClient


async def run_checks(checks=None):
    install_hooks()
    checks = [] if checks is None else checks

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(name + ": " + str(detail))

    # Dummy key and endpoint; even a fixture bug cannot send a real credential.
    profile = {"provider": "openai", "model": "gpt-5.6-sol", "api_key": "offline-fixture",
               "api_style": "chat_completions", "base_url": "https://offline.invalid/v1",
               "context_window": 1000000}
    profiles = {"openai": profile}
    saved = {"default_provider": "openai", "providers": [profile],
             "reflection": {"enabled": True, "inherit_main": True},
             "maintenance": {"enabled": True, "inherit_main": True}}
    fixtures = {"relation": "replacement", "missing_usage": False}
    wire = []

    def handler(request):
        body = json.loads(request.content)
        wire.append(body)
        messages = body["messages"]
        prompt = str(messages[0].get("content", ""))
        content, calls = None, []
        if prompt.startswith("Extract only stable"):
            sources = json.loads(prompt[prompt.index('{"sources":'):])["sources"]
            user = next(s for s in sources if s["role"] == "user")
            value = "Quartz" if "Quartz" in user["text"] else "Zephyr"
            content = dumps({"memories": [{"title": "Alder database", "summary": "Alder storage decision",
                "content": f"Project Alder uses {value} as its database engine.", "confidence": .94,
                "evidence": [user["text"]], "reason": "Explicit durable decision",
                "fact": {"subject": "user", "topic": "Alder", "attribute": "database",
                         "value": value, "kind": "historical" if fixtures.get("misclassify") else "current", "atomic": True,
                         "attribution": {"origin": "user", "adoption": "accepted",
                                         "provenance": [{"source_id": user["id"], "quote": user["text"]}]}},
                "provenance": [{"source_id": user["id"], "quote": user["text"]}]}],
                "core_updates": {"diagnostic": "CORE_DIAGNOSTIC_ONLY"}})
            if user["text"] == "Hi":
                content = dumps({"memories": [], "skip_reason": "Greeting without a durable assertion"})
            if fixtures.get("malformed_extraction"):
                content = '{"memories":[{"title":"broken"}}]}'
        elif prompt.startswith("Repair one long-term memory extraction"):
            data = json.loads(prompt[prompt.index('{"sources":'):])
            item = copy.deepcopy(data["original_memory"])
            item["fact"]["kind"] = "current"
            user = next(s for s in data["sources"] if s["role"] == "user" and not s.get("context_only"))
            proof = {"source_id":user["id"],"quote":user["text"]}
            item["provenance"] = [proof]
            content = dumps({"memory":item,"field_evidence":{"kind":proof},"reason":"Explicit established state"})
        elif prompt.startswith(protocol.RELATION_PROMPT):
            data = json.loads(prompt[prompt.index('{"incoming":'):])
            content = dumps({"relations": [{"id": old["id"], "relation": fixtures["relation"],
                "confidence": .94, "same_scope": True, "covers_entire_record": True,
                "covers_incoming": True, "explicit_update": True,
                "incoming_quote": protocol.source_texts(data["incoming"])[0],
                "existing_quote": old["content"], "reason": "Controlled grounded decision"}
                for old in data["candidates"]]}) if fixtures["relation"] != "invalid" else "{}"
        elif prompt.startswith("Memory capacity is full."):
            content = '{"decisions":[]}'
        else:
            tool_messages = [m for m in messages if m["role"] == "tool"]
            if fixtures.get("direct_answer"):
                content = '{"answer":"Quartz","reason":"Read the available evidence"}'
            elif len(tool_messages) == 0:
                calls = [{"id": "offline-catalog", "type": "function", "function": {
                    "name": "tool_catalog", "arguments": '{"query":"memory_read"}'}}]
            elif len(tool_messages) == 1:
                calls = [{"id": "offline-read", "type": "function", "function": {
                    "name": "memory_read", "arguments": dumps({"memory_id": fixtures["read_id"]})}}]
            else:
                content = '{"answer":"Quartz"}'
        usage = {} if fixtures["missing_usage"] else {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        if body.get("stream"):
            delta = {"content": content} if not calls else {"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}
            chunks = [{"model": profile["model"], "choices": [{"index": 0, "delta": delta}], "usage": usage}]
            sse = "".join("data: " + dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"model": profile["model"], "choices": [{"message": {
            "role": "assistant", "content": content, "tool_calls": calls}}], "usage": usage})

    def mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return ObservedHTTPClient(*args, **kwargs)

    apps = []
    clock = CLOCK.set(datetime(2026, 1, 1, tzinfo=UTC))
    mock_token = OFFLINE_MOCK.set(True)
    try:
        with offline_network_guard() as attempts, tempfile.TemporaryDirectory(prefix="memprobe-adapter-") as folder:
            root = Path(folder)
            with patch.object(httpx, "AsyncClient", mock_client):
                checks.extend(await route_checks())
                check("native_memory_protocol_inherited", BenchMemory._adjudicate is MemoryService._adjudicate
                      and BenchMemory.reflect is MemoryService.reflect and BenchMemory.agent_record is MemoryService.agent_record)
                check("native_recovery_and_read_paths_inherited", BenchMemory.recover_assessment is MemoryService.recover_assessment
                      and BenchMemory.agent_views is MemoryService.agent_views and BenchMemory.create is MemoryService.create
                      and BenchMemory.should_reflect is MemoryService.should_reflect)
                app = BenchApp(isolated_settings(root / "ingestion"), saved, profiles)
                apps.append(app)
                await app.initialize_benchmark()
                try:
                    units, orphan = replay_units({"sessions": [{"dialogue": [
                        {"role": "assistant", "content": "orphan"},
                        {"role": "user", "content": "remember durable fact"},
                        {"role": "assistant", "content": "intermediate"},
                        {"role": "tool_output", "content": "TOOL_ONLY_SENTINEL"},
                        {"role": "assistant", "content": "final"},
                        {"role": "user", "content": "unanswered"}]}]})
                    check("native_role_mapping", len(units) == 2 and len(orphan) == 1 and units[0]["assistant"] == "final")
                    check("tool_message_audit_only", "TOOL_ONLY_SENTINEL" not in units[0]["user"] + units[0]["assistant"])
                    for i, unit in enumerate(({"user": "Hi", "assistant": "Hello"}, {"user": "remember this", "assistant": ""})):
                        r = await ingest_unit(app, {"unit_id": f"skip{i}", **unit}, root / f"skip{i}", "synthetic", profiles, i)
                        # Every completed turn is inspected; only a missing
                        # final reply bypasses post-run extraction. The model's
                        # durability filter, not this gate, decides whether a
                        # memory is written.
                        expected = (r["reflection_triggered"] and bool(r["calls"])) if i == 0 else (not r["reflection_triggered"] and not r["calls"])
                        check(f"native_turn_inspection_{i}", r["status"] == "completed" and expected)
                    check("untriggered_messages_preserved_for_later_context", bool(await app.db.fetchone("SELECT id FROM messages WHERE content='Hi'")))
                    results = []
                    for i, value in enumerate(("Zephyr", "Quartz", "Zephyr")):
                        fixtures["relation"] = "keep_old" if i == 2 else "replacement"
                        unit = {"unit_id": f"u{i}", "user": f"Please remember my approved decision: Project Alder uses {value} as its database engine.",
                                "assistant": "I recorded this durable project architecture decision."}
                        r = await ingest_unit(app, unit, root / f"u{i}", "synthetic", profiles, i)
                        results.append(r)
                        check(f"native_ingestion_{i}", r["status"] == "completed", r.get("error"))
                    memories = [json.loads(x["data_json"]) for x in await app.db.fetchall("SELECT data_json FROM memories")]
                    current = [m for m in memories if protocol.usage_state(m) == "current"]
                    check("automatic_grounded_update", len(current) == 1 and current[0]["fact"]["value"] == "Quartz", [protocol.usage_state(m) for m in memories])
                    check("historical_and_not_adopted_retained", {protocol.usage_state(m) for m in memories} == {"current", "historical", "not_adopted"})
                    check("backend_provenance_bound", all(p.get("message_id") and p.get("role") == "user" and p.get("observed_at", "").startswith("2026-01-01") for m in memories for p in m["provenance"]))
                    check("native_fact_fields_retained", all(m["fact"]["subject"] == "user" and m["fact"]["topic"] == "Alder" and m["fact"]["attribute"] == "database" for m in memories))
                    check("automatic_is_not_user_approval", all(m.get("acceptance_source") == "automatic" and not m.get("user_confirmed") for m in memories))
                    check("native_bounded_previous_turn_context", any(s.get("context_only") and "Zephyr" in s["text"] for s in results[1]["source_records"]))
                    boundary = await ingest_unit(app, {"unit_id": "boundary", "session": 99, "user": "Hi again", "assistant": "Hello"},
                                                 root / "boundary", "synthetic", profiles, 4)
                    check("session_boundary_not_flattened", not any(s.get("context_only") for s in boundary["source_records"]))
                    message_count = (await app.db.fetchone("SELECT count(*) AS n FROM messages"))["n"]
                    replay_unit = results[0]["input"] | {"unit_id": "u0"}
                    original_run, original_sources = await replay_source_records(app, replay_unit, "synthetic")
                    replay_clock = CLOCK.set(datetime(2026, 1, 2, tzinfo=UTC))
                    try:
                        resumed_run, resumed_sources = await replay_source_records(app, replay_unit, "synthetic")
                    finally:
                        CLOCK.reset(replay_clock)
                    check("replay_materialization_is_idempotent", (await app.db.fetchone("SELECT count(*) AS n FROM messages"))["n"] == message_count)
                    check("replay_preserves_original_evidence_time", original_run == resumed_run and original_sources == resumed_sources)
                    calls = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("u*/model_calls/*.json")]
                    relation_calls = [c for c in calls if c["phase"] == "relationship"]
                    check("native_relation_model_observed", len(relation_calls) == 2 and all(c["transport"]["effective_timeout"]["read"] == 300 for c in relation_calls))
                    check("native_extraction_http_timeout", all(c["transport"]["effective_timeout"]["read"] == 120 for c in calls if c["phase"] == "extraction"))
                    check("wire_model_and_sampling_truthful", all(c["transport"]["wire_model"] == "gpt-5.6-sol" and c["transport"]["wire_parameters"] == {} for c in calls))
                    check("mock_usage_explicitly_labeled", all(c["offline_mock_transport"] and c["usage"]["input_tokens"] == 100 for c in calls))
                    before = len(wire)
                    again = await ingest_unit(app, results[0]["input"] | {"unit_id": "u0"}, root / "u0", "synthetic", profiles, 0)
                    check("completed_unit_not_reexecuted", len(wire) == before and again == results[0])
                    # Native incomplete decision is preserved; never silently approve it.
                    fixtures["relation"] = "invalid"
                    failed = await ingest_unit(app, {"unit_id": "incomplete", "user": "Remember: Project Alder uses Zephyr.", "assistant": "A durable decision."}, root / "incomplete", "synthetic", profiles, 3)
                    check("invalid_decision_kept_incomplete", failed["status"] == "completed" and failed["memory_states"][0]["decision_state"] == "incomplete")
                    await seed(app, "dispute", title="Alder disagreement", summary="Alder unresolved",
                               content="DISPUTED_BODY_ONLY", decision_state="unresolved", conflict_status="needs_review", conflicts=[current[0]["id"]])
                    await seed(app, "plan", title="Alder future plan", summary="Alder future plan",
                               content="FUTURE_PLAN_ONLY", fact={"kind": "plan", "subject": "user", "topic": "Alder", "attribute": "database"})
                    candidates = await app.memory_service.relevant_items("Alder database")
                    notices = await app.memory_service.context_notices("Alder database")
                    check("top_five_and_status_notices", 0 < len(candidates) <= 5 and all(BenchMemory._eligible_for_context(m) for m in candidates) and bool(notices))
                    snap = root / "snapshot.db"
                    snapshot_database(app.settings.database, snap)
                    fingerprint = memory_fingerprint(snap)
                    payloads, outcomes = {}, {}
                    fixtures["read_id"] = current[0]["id"]
                    for arm in ("A", "B"):
                        settings = isolated_settings(root / arm)
                        settings.ensure_dirs()
                        snapshot_database(snap, settings.database)
                        trial = BenchApp(settings, saved, profiles, frozen=True, arm=arm, answer_schema=schema("current_value"))
                        apps.append(trial)
                        await trial.initialize_benchmark()
                        trial.fixed_candidates, trial.fixed_notices = copy.deepcopy(candidates), copy.deepcopy(notices)
                        check(arm + "_same_initial_memory", memory_fingerprint(settings.database) == fingerprint)
                        out = await answer_question(trial, "What database does Project Alder use?", root / arm / "trial", "synthetic", 0, profiles)
                        outcomes[arm] = out
                        check(arm + "_real_runtime_with_tool_read", out["status"] == "completed" and out["answer"] == '{"answer":"Quartz"}', out.get("error"))
                        tool_names = [t["name"] for t in out["tool_observations"]]
                        check(arm + "_tool_chain", tool_names[:2] == ["tool_catalog", "memory_read"]
                              and tool_names.count("memory_search") == 0 and 2 <= tool_names.count("memory_read") <= 6, out["tool_observations"])
                        read_ids = {t["arguments"].get("memory_id") for t in out["tool_observations"] if t["name"] == "memory_read"}
                        check(arm + "_update_notices_verified_despite_confident_answer", {"dispute","plan"} <= read_ids)
                        raw_calls = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (root / arm / "trial/model_calls").glob("*.json")), key=lambda c: c["started_at"])
                        payloads[arm] = raw_calls[0]["request"]
                        check(arm + "_cumulative_usage", len(raw_calls) == 4 and sum(c["usage"]["input_tokens"] for c in raw_calls) == 400)
                        context_events = [e["context"] for e in out["events"] if e["type"] == "context_prepared"]
                        check(arm + "_metadata_estimate_separate_from_api_usage", context_events[0]["memory_context"]["structured_metadata_tokens_estimated"] > 0)
                        check(arm + "_no_external_answer_source", all([m["content"] for m in c["request"]["messages"] if m["role"] == "user"] == ["What database does Project Alder use?"] for c in raw_calls)
                              and "CORE_DIAGNOSTIC_ONLY" not in dumps(raw_calls))
                        listing = await trial.memory_service.list()
                        check(arm + "_core_diagnostics_hidden", listing["core_entries"] == [] and bool(await trial.memory_service.core_entries_detailed()))
                    a_text, b_text = dumps(payloads["A"]), dumps(payloads["B"])
                    check("only_candidate_presentation_differs", a_text.replace(current[0]["content"], current[0]["summary"]) == b_text)
                    check("paired_notices_preserved", outcomes["A"]["memory_notices"] == outcomes["B"]["memory_notices"] == notices and "Memory status notices" in a_text)
                    check("snapshot_not_reinforced_by_answers", memory_fingerprint(snap) == fingerprint)
                    fixtures["direct_answer"] = True
                    checked = await answer_question(apps[-1], "Who reported the Alder database change?", root / "bounded-verification", "synthetic", 9, profiles)
                    names = [x["name"] for x in checked["tool_observations"]]
                    check("runtime_forces_one_bounded_read_verification", checked["status"] == "completed"
                          and names.count("memory_search") <= 1 and 1 <= names.count("memory_read") <= 5, names)
                    check("known_source_candidate_read_without_redundant_search", names.count("memory_search") == 0, names)
                    verify_calls = list((root / "bounded-verification/model_calls").glob("*.json"))
                    check("verification_usage_is_in_answer_ledger", len(verify_calls) == 2)
                    checked_run = checked["response"]["run"]
                    verification_events = [e["tool_result"] for e in checked["events"]
                                           if e["type"] == "tool_result" and e.get("origin") == "memory_verification"]
                    check("verification_uses_standard_tool_event_schema", len(verification_events) == len(names)
                          and all(e.get("tool_call_id") and e.get("success") and "output" in e and "duration_ms" in e for e in verification_events))
                    evidence = await apps[-1].evidence.search("", run_id=checked_run["id"])
                    check("verification_saved_in_evidence_store", len(evidence) == len(names)
                          and all(e["data"].get("origin") == "memory_verification" for e in evidence))
                    checkpoint = await apps[-1].checkpoints.get(checked_run["id"])
                    check("verification_checkpoint_retains_results_and_limit", checkpoint["state"]["memory_verification_done"]
                          and any(m.get("name") == "memory_read" for m in checkpoint["state"]["model_messages"]))
                    # Simulate compaction removing the marker and tool messages;
                    # the persisted run flag must still prevent another pass.
                    recovery = copy.deepcopy(checkpoint["state"])
                    recovery["model_messages"] = [m for m in recovery["model_messages"]
                        if m.get("role") in {"system", "user"} and not str(m.get("content", "")).startswith("Memory verification (runtime)")]
                    apps[-1].tool_observations = []
                    ctx = call_context(root / "verification-resume", "synthetic", "answer", profiles)
                    ct = CALL_CONTEXT.set(ctx)
                    try:
                        recovered_run = await apps[-1]._new_run(checked_run["conversation_id"], "Who reported the Alder database change?", "normal")
                        resumed = await apps[-1]._execute_run(recovered_run, recovery)
                    finally:
                        CALL_CONTEXT.reset(ct)
                    check("verification_not_repeated_after_compaction_and_resume", recovered_run["status"] == "completed" and not resumed.get("error")
                          and not apps[-1].tool_observations and len(ctx["calls"]) == 1)
                    privacy = await apps[-1].db.get_json("privacy_settings", {})
                    await apps[-1].db.set_json("privacy_settings", {**privacy, "memory_in_context_enabled": False})
                    disabled = await answer_question(apps[-1], "Who reported the Alder database change?", root / "verification-disabled", "synthetic", 10, profiles)
                    check("automatic_verification_respects_injection_toggle", disabled["status"] == "completed" and not disabled["tool_observations"])
                    await apps[-1].db.set_json("privacy_settings", privacy)
                    from dataclasses import replace
                    original_settings = apps[-1].settings
                    apps[-1].settings = replace(original_settings, max_steps=1)
                    limited = await answer_question(apps[-1], "Who reported the Alder database change?", root / "verification-step-limit", "synthetic", 11, profiles)
                    check("automatic_verification_respects_step_limit", limited["status"] == "completed" and not limited["tool_observations"])
                    apps[-1].settings = original_settings
                    fixtures.pop("direct_answer")
                    await apps[-2].db.set_json("isolated_a_write", True)
                    check("arm_writes_isolated", await apps[-1].db.get_json("isolated_a_write", None) is None)
                    # Check actual tool routes for every supported interpretation.
                    ex = ReadOnlyMemoryTools(app.settings, app._request_approval, app.publish_artifact, app.remember, app.mcp_manager, app)
                    run = await app._new_run(None, "Offline access fixture", "normal")
                    await ex.execute(run["id"], "tool_catalog", {"query": "memory"}, "catalog")
                    for m in memories:
                        state = protocol.usage_state(m)
                        purpose = "current" if state == "current" else "history"
                        viewed = json.loads(await ex.execute(run["id"], "memory_read", {"memory_id": m["id"], "purpose": purpose}, m["id"]))["memory"]
                        check("tool_read_" + state, viewed.get("content") == m["content"] and viewed["usage_state"] == state and bool(viewed["usage_guidance"]))
                    for mid, purpose, state in (("dispute", "disputes", "disputed"), ("plan", "qualified", "qualified")):
                        viewed = json.loads(await ex.execute(run["id"], "memory_read", {"memory_id": mid, "purpose": purpose}, mid))["memory"]
                        check("tool_read_" + state, viewed["usage_state"] == state and not viewed.get("content_withheld"))
                    for mid, extra in (("pending", {"status": "pending_review"}), ("rejected", {"acceptance_source": "rejected"}), ("private", {"sensitive": True})):
                        await seed(app, mid, **extra)
                        try:
                            viewed = json.loads(await ex.execute(run["id"], "memory_read", {"memory_id": mid, "purpose": "all"}, mid))["memory"]
                        except ValueError as exc:
                            check("read_gate_" + mid, mid == "private" and "include_sensitive" in str(exc))
                        else:
                            check("read_gate_" + mid, viewed.get("content_withheld") and "content" not in viewed)
                    # Capture maintenance requests and native configured relation overrides.
                    await app.db.set_json("memory_policy", {"max_active": 1, "relation_http_read_seconds": 77, "relation_timeout_seconds": 111})
                    ctx = call_context(root / "auxiliary", "synthetic", "extraction", profiles)
                    ct = CALL_CONTEXT.set(ctx)
                    try:
                        async def model(prompt):
                            return (await ObservedModel(app.settings, profile).complete([{"role": "user", "content": prompt}], [], [])).content
                        await app.memory_service.maintain(model)
                        relation_prompt = relation_calls[0]["request"]["messages"][0]["content"]
                        await app._memory_relation_model(relation_prompt)
                        check("native_maintenance_and_policy_override", [c["phase"] for c in ctx["calls"]] == ["maintenance", "relationship"] and ctx["calls"][-1]["transport"]["effective_timeout"]["read"] == 77)
                        fixtures["missing_usage"] = True
                        await model('Extract only stable\n{"sources":[{"id":"fixture","role":"user","text":"Zephyr"}]}')
                        check("missing_usage_detected", len(ctx["accounting_errors"]) == 1)
                    finally:
                        CALL_CONTEXT.reset(ct)
                        fixtures["missing_usage"] = False
                    # A context object alone must not authorize live API calls.
                    ct = CALL_CONTEXT.set(call_context(root / "guard", "synthetic", "answer", profiles, offline_mock_transport=False))
                    try:
                        with patch.dict("os.environ", {"KORE_MEMPROBE_ALLOW_LIVE": "0"}):
                            try:
                                await ObservedModel(app.settings, profile).complete([], [], [])
                            except RuntimeError as exc:
                                check("paid_calls_require_explicit_live_mode", "disabled" in str(exc))
                            else:
                                check("paid_calls_require_explicit_live_mode", False)
                    finally:
                        CALL_CONTEXT.reset(ct)
                    check("clock_shared_by_all_memory_layers", memory_module.now() == protocol.now() == application_module.now() == memory_now())
                    async def clock_task(minutes):
                        t = CLOCK.set(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes))
                        try:
                            await asyncio.sleep(0)
                            return protocol.datetime.now(UTC).minute
                        finally:
                            CLOCK.reset(t)
                    check("concurrent_scenes_have_isolated_clocks", await asyncio.gather(clock_task(7), clock_task(9)) == [7, 9])
                    # Exercise the ingestion entry point at the old and new
                    # boundaries, retaining the native decay implementation.
                    cadence_app = BenchApp(isolated_settings(root / "cadence"), saved, profiles)
                    apps.append(cadence_app)
                    await cadence_app.initialize_benchmark()
                    stamp = "2026-01-01T00:00:00Z"
                    await seed(cadence_app, "cadence-fact", created_at=stamp, updated_at=stamp,
                               observed_at=stamp, last_accessed_at=stamp)
                    with patch.object(cadence_app.memory_service, "maintain_decay", wraps=cadence_app.memory_service.maintain_decay) as decay, \
                         patch.object(cadence_app.memory_service, "maintain", wraps=cadence_app.memory_service.maintain) as capacity:
                        for index, expected in ((0, 0), (14, 0), (358, 0), (359, 1), (360, 1), (718, 1), (719, 2), (720, 2)):
                            unit = {"unit_id": f"u{index}", "user": "Hi", "assistant": "Hello"}
                            outcome = await ingest_unit(cadence_app, unit, root / "cadence" / str(index), "cadence", profiles, index)
                            check(f"decay_boundary_turn_{index + 1}", outcome["status"] == "completed" and decay.await_count == expected, outcome.get("error"))
                            if index in (359, 719):
                                hour = 6 if index == 359 else 12
                                expected_at = f"2026-01-01T{hour:02d}:00:00Z"
                                fact = await cadence_app.memory_service.get("cadence-fact", touch=False)
                                check(f"decay_clock_turn_{index + 1}", outcome["periodic_decay"]["at"] == expected_at
                                      and fact["last_decay_calculated_at"] == expected_at and fact["decay_score"] < .45)
                                replay = await ingest_unit(cadence_app, unit, root / "cadence" / str(index), "cadence", profiles, index)
                                check(f"cached_turn_{index + 1}_does_not_repeat_decay", replay == outcome and decay.await_count == expected)
                        check("per_turn_capacity_maintenance_preserved", capacity.await_count == 8)
                        cadence_calls = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "cadence").glob("*/model_calls/*.json")]
                        check("cadence_checks_inspect_completed_turns_with_mock_calls_only",
                              len(cadence_calls) == 8 and all(c.get("offline_mock_transport") and c['phase'] == 'extraction' for c in cadence_calls))
                    recovery_app = BenchApp(isolated_settings(root / "recovery"), saved, profiles)
                    apps.append(recovery_app)
                    await recovery_app.initialize_benchmark()
                    fixtures["missing_usage"] = False
                    fixtures["relation"] = "replacement"
                    seeded = await ingest_unit(recovery_app, {"unit_id": "base", "user": "Remember: Project Alder uses Zephyr.",
                                               "assistant": "Noted."}, root / "recovery/base", "recovery", profiles, 0)
                    native_relation = recovery_app._memory_relation_model
                    attempts_relation = []
                    async def transient_relation(prompt):
                        attempts_relation.append(prompt)
                        if len(attempts_relation) == 1:
                            raise protocol.RelationUnavailable("offline_injected_transient")
                        return await native_relation(prompt)
                    with patch.object(recovery_app, "_memory_relation_model", transient_relation):
                        recovered = await ingest_unit(recovery_app, {"unit_id": "update", "user": "The module now uses Quartz.",
                                                      "assistant": "Noted."}, root / "recovery/update", "recovery", profiles, 1)
                    check("short_update_uses_native_reflection", recovered["reflection_triggered"])
                    check("native_relation_recovery_does_not_falsely_block_adapter", recovered["status"] == "completed"
                          and recovered.get("native_relationship_recovery_completed") and len(attempts_relation) == 2, recovered.get("error"))
                    recovered_memory = await recovery_app.memory_service.get(recovered["records_created_or_updated"][0], touch=False)
                    check("native_retry_is_bounded_and_source_bound", recovered_memory["auto_reassessment_count"] == 1
                          and protocol.usage_state(recovered_memory) == "current" and bool(recovered_memory["provenance"]))
                    fixtures["misclassify"] = True
                    repaired = await ingest_unit(recovery_app, {"unit_id":"semantic", "user":"Remember: Project Alder database uses Zephyr in production.",
                                                 "assistant":"Noted."}, root / "recovery/semantic", "recovery", profiles, 2)
                    fixtures.pop("misclassify")
                    repaired_memory = await recovery_app.memory_service.get(repaired["records_created_or_updated"][0],touch=False)
                    check("native_reextraction_then_adjudication_in_adapter", repaired["status"] == "completed"
                          and protocol.usage_state(repaired_memory) == "current" and bool(repaired_memory.get("extraction_correction")))
                    semantic_calls = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "recovery/semantic/model_calls").glob("*.json")]
                    phases = [c["phase"] for c in semantic_calls]
                    check("shared_correction_cycle_usage_fully_recorded", phases.count("extraction") == 2 and phases.count("relationship") == 2
                          and sum(c["usage"]["input_tokens"] for c in semantic_calls) == 400)
                    # Product _post_run logs a malformed extraction and returns;
                    # the adapter must retain this failure, not retry or stop all
                    # later history. Verify with real reflection and mock HTTP.
                    fixtures["malformed_extraction"] = True
                    failure_unit = {"unit_id":"invalid", "user":"Remember this durable project decision.", "assistant":"Noted."}
                    rows_before = await recovery_app.db.fetchall("SELECT id,data_json FROM memories ORDER BY id")
                    with patch.object(recovery_app, "_event") as events, \
                         patch.object(recovery_app.memory_service, "maintain", wraps=recovery_app.memory_service.maintain) as maintenance:
                        native_ctx = call_context(root/"recovery/native-failure", "recovery", "extraction", profiles)
                        native_token = CALL_CONTEXT.set(native_ctx)
                        native_wire_before = len(wire)
                        try:
                            await recovery_app._post_run({"id":"native-failure", "conversation_id":"native-failure", "user_message":failure_unit["user"]},
                                                        {"final_message":{"content":"Noted."}})
                        finally:
                            CALL_CONTEXT.reset(native_token)
                        check("native_malformed_extraction_records_post_run_failure",
                              events.await_args_list[-1].args[1] == "post_run_failed" and maintenance.await_count == 0
                              and events.await_args_list[-1].kwargs["error"]["message"] == "Invalid memory extraction result"
                              and len(wire) == native_wire_before + 1 and len(native_ctx["calls"]) == 1)
                    with patch.object(recovery_app.memory_service, "maintain", wraps=recovery_app.memory_service.maintain) as maintenance:
                        failed = await ingest_unit(recovery_app, failure_unit, root/"recovery/invalid", "recovery", profiles, 3)
                        check("malformed_extraction_retained_as_product_failure",
                              failed["status"] == "product_failure" and failed["failure_scope"] == "native_post_run"
                              and failed["memory_processing_succeeded"] is False and bool(failed["source_records"]))
                        check("malformed_extraction_no_retry_or_maintenance", len(failed["calls"]) == 1 and maintenance.await_count == 0)
                    check("malformed_extraction_no_memory_write", rows_before == await recovery_app.db.fetchall("SELECT id,data_json FROM memories ORDER BY id"))
                    count_before = len(wire)
                    replayed = await ingest_unit(recovery_app, failure_unit, root/"recovery/invalid", "recovery", profiles, 3)
                    check("failed_product_turn_resume_is_idempotent", replayed == failed and len(wire) == count_before)
                    fixtures["missing_usage"] = True
                    accounting_failed = await ingest_unit(recovery_app, {**failure_unit,"unit_id":"accounting-failure"},
                                                          root/"recovery/accounting-failure", "recovery", profiles, 4)
                    check("missing_usage_still_blocks_malformed_result", accounting_failed["status"] == "error")
                    fixtures["missing_usage"] = False
                    fixtures.pop("malformed_extraction")
                    with patch.object(recovery_app.memory_service, "reflect", side_effect=ValueError("unrecognized integrity failure")):
                        unknown = await ingest_unit(recovery_app, {**failure_unit,"unit_id":"unknown-failure"},
                                                    root/"recovery/unknown-failure", "recovery", profiles, 5)
                    check("unknown_failure_not_silently_skipped", unknown["status"] == "error")
                    probe = {"question": "synthetic", "probe_type": "current_value", "gold_answer": "Quartz"}
                    check("scorer_correct_and_incorrect", grade('{"answer":"Quartz"}', probe, {})["match"] == "correct" and grade('{"answer":"Zephyr"}', probe, {})["match"] == "incorrect")
                    check("empty_error_not_correct", all(grade(text, probe, {})["match"] == "execution_error" for text in ("", "[ERROR: timeout]")))
                    from runner import validate_preflights
                    test_config = {"fixture": "no official answers"}
                    from common import digest
                    for name in ("offline_preflight.json", "live_preflight.json"):
                        dump(root / "gate" / name, {"status": "passed", "configuration_fingerprint": digest(test_config)})
                    validate_preflights(test_config, root / "gate")
                    check("matching_preflight_identity_accepted", True)
                    for name, directory in (("stale_preflight_rejected", root / "gate"), ("missing_preflight_rejected", root / "missing")):
                        try:
                            validate_preflights({"fixture": "changed"}, directory)
                        except RuntimeError:
                            check(name, True)
                        else:
                            check(name, False)
                    try:
                        dump(INITIAL_RUN / "must-not-write.json", {})
                    except ValueError:
                        check("initial_run_write_protection", True)
                    else:
                        check("initial_run_write_protection", False)
                    # Exercise reset only inside the verified temporary sandbox.
                    assert recovery_app.settings.data_dir.resolve().is_relative_to(root.resolve())
                    assert recovery_app.settings.user_skills_dir.resolve().is_relative_to(root.resolve())
                    with patch.object(recovery_app.connector_service, "revoke_all", return_value={}):
                        cleared = await recovery_app.data_delete_all({"include_workspace":False})
                    check("reset_clears_new_source_index", cleared["deleted"] and not await recovery_app.db.fetchall("SELECT memory_id FROM memory_source_fts"))
                    check("no_network_attempts", not attempts)
                finally:
                    for instance in reversed(apps):
                        await close(instance)
                    apps.clear()
    finally:
        for instance in reversed(apps):
            await close(instance)
        CLOCK.reset(clock)
        OFFLINE_MOCK.reset(mock_token)
    return checks
