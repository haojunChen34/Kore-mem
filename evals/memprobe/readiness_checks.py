"""Offline launch contracts for the current product memory implementation.

No benchmark answers, saved user data, or model requests are used here.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from common import offline_network_guard
from check_memory_search import isolated_settings, close
from kore_next import application as application_module
from kore_next import memory as memory_module
from kore_next import memory_quality as quality
from kore_next.application import Application
from kore_next.memory import MemoryService
from kore_next.memory_protocol import usage_state, partial_edit_constraints
from kore_next.util import dumps
from runtime import BenchApp, BenchMemory, DECAY_INTERVAL_TURNS


async def run_checks(checks=None):
    checks = checks if checks is not None else []

    def check(name, value):
        checks.append({"name": "readiness_" + name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    for name in ("should_reflect", "reflect", "create", "_adjudicate", "_replace_record", "_apply_decided", "recover_assessment",
                 "agent_record", "agent_views", "agent_search", "maintain", "maintain_decay"):
        check("inherits_memory_" + name, getattr(BenchMemory, name) is getattr(MemoryService, name))
    for name in ("_execute_run", "_memory_relation_model", "_memory_source_records", "tool_call"):
        check("inherits_application_" + name, getattr(BenchApp, name) is getattr(Application, name))
    check("answer_gate_uses_current_product_helper",
          application_module.needs_memory_verification is quality.needs_memory_verification)
    check("answer_evidence_uses_current_product_helper",
          application_module.verify_memory_answer is quality.verify_memory_answer)
    check("decay_keeps_360_turn_policy", DECAY_INTERVAL_TURNS == 360)
    check("extraction_review_uses_current_product_helper",
          memory_module.extraction_issues is quality.extraction_issues)
    check("initial_gate_accepts_completed_short_turn", BenchMemory.should_reflect("否决了。", "收到。"))
    check("initial_gate_requires_user_and_final_reply",
          not BenchMemory.should_reflect("", "Done") and not BenchMemory.should_reflect("Changed", ""))
    for index, topic in enumerate(("gym", "database", "insurance provider", "newsletter", "seed supplier", "quux-17")):
        text = f"I switched to another {topic}."
        check(f"topic_independent_initial_trigger_{index}", BenchMemory.should_reflect(text, "Understood."))
        check(f"topic_independent_transient_inspection_{index}",
              BenchMemory.should_reflect(text[:-1] + " temporarily for this test.", "Understood."))
    for index, topic in enumerate(("newsletter", "insurance", "archive retention", "unknown-topic-42", "灌溉周期")):
        text = f"Please remember: I switched my {topic} to option-b."
        sources = [{"id": "u", "role": "user", "text": text}]
        check(f"topic_independent_empty_review_{index}",
              quality.extraction_issues({"memories": []}, sources) ==
              ["assertion_or_retention_request_with_empty_extraction"])
        missing = {"content": text, "fact": {"subject": "user", "topic": topic, "kind": "current"}}
        check(f"topic_independent_missing_evidence_{index}",
              "item:0:missing_current_user_evidence" in quality.extraction_issues({"memories": [missing]}, sources))
    for index, subject in enumerate(("Elena", "elena", "小周", "人物-42")):
        text = "I now prefer morning appointments."
        item = {"content": text, "fact": {"subject": subject}, "provenance": [{"source_id": "u", "quote": text}]}
        check(f"subject_guard_independent_of_name_shape_{index}",
              quality.subject_evidence_conflict(item, [{"id": "u", "role": "user", "text": text}]))
    old = {"id":"old", "content":"Earlier configuration", "usage_state":"historical",
           "fact":{"subject":"user's home scanner", "topic":"image capture", "attribute":"format", "value":"BMP", "kind":"current"}}
    new = {**old, "id":"new", "usage_state":"current", "supersedes":["old"],
           "fact":{**old["fact"], "value":"PNG"}}
    messages = [{"role":"tool", "name":"memory_read", "content":dumps({"memory":m})} for m in (old,new)]
    plan = quality.memory_verification_plan("Did my scanner format change?", "PNG", messages, candidates=[new])
    check("owned_asset_scope_uses_real_read_evidence", not plan["needed"] and not plan["ambiguous"])
    negative = {**new, "fact":{**new["fact"], "value":"no longer BMP"}, "supersedes":[]}
    plan = quality.memory_verification_plan("What is my scanner format now?", "No longer BMP", [], candidates=[negative])
    check("negation_does_not_complete_open_value_query", "current_value" in plan["evidence_gaps"])
    text = "I now prefer morning appointments."
    item = {"fact":{"subject":"Elena"}, "provenance":[{"source_id":"u","quote":text}]}
    check("subject_conflict_uses_current_source", quality.subject_evidence_conflict(item,[{"id":"u","role":"user","text":text}]))
    check("quoted_first_person_is_not_the_requester", not quality.subject_evidence_conflict(item,[{"id":"u","role":"user","text":'She wrote: "'+text+'"'}]))
    plan = quality.memory_verification_plan("告诉我当前的导出格式。", "No longer BMP", [], candidates=[negative])
    check("imperative_value_request_checks_negative_draft", plan["answer_incomplete"] and plan["needed"])
    plan = quality.memory_verification_plan("Tell me my current scanner format.", "No longer BMP", messages, candidates=[new])
    check("read_evidence_does_not_hide_incomplete_draft", plan["answer_incomplete"] and plan["needed"] and not plan["missing"])
    renamed = {**new, "fact":{**new["fact"], "topic":"document digitization"}}
    check("replacement_edge_supports_renamed_topic", quality._same_focus(old, renamed))
    check("project_identity_blocks_cross_scope_edge", not quality._same_focus({**old,"project_id":"a"}, {**renamed,"project_id":"b"}))

    with offline_network_guard() as network_attempts, tempfile.TemporaryDirectory(prefix="memprobe-readiness-") as folder:
        app = BenchApp(isolated_settings(Path(folder)), {}, {}, frozen=False, arm="native")
        sequence = 0
        try:
            await app.initialize_benchmark()
            privacy = await app.db.get_json("privacy_settings")
            check("ingestion_automatic_save_and_injection", privacy["auto_memory_enabled"] and
                  privacy["memory_review_mode"] == "automatic" and privacy["memory_in_context_enabled"])
            check("native_candidates_not_overridden", app.fixed_candidates is None and app.fixed_notices is None)

            async def create(text, *, fact=None, target=None, fragment=None):
                nonlocal sequence
                sequence += 1
                sid = "readiness-source-" + str(sequence)

                async def decision(prompt):
                    payload = json.loads(prompt[prompt.index('{"incoming":'):])
                    entries = []
                    for old in payload["candidates"]:
                        replace = old["id"] == target
                        entries.append({"id": old["id"], "relation": "replacement" if replace else "coexist",
                            "confidence": .95, "same_scope": True, "covers_entire_record": not (replace and fragment),
                            "covers_incoming": True, "explicit_update": True, "incoming_quote": text,
                            "existing_quote": old["content"], "partial_update_safe": True,
                            "superseded_fragments": [fragment] if replace and fragment else [],
                            "reason": "Synthetic bounded contract decision; no model request"})
                    return dumps({"relations": entries})

                return await app.memory_service.create(title="Cedar project database", summary=text, content=text,
                    confidence=.9, fact={"subject": "Cedar project", "topic": "storage", "attribute": "database",
                                         "kind": "current", **(fact or {})},
                    source_type="conversation", source_id=sid, provenance=[{"source_id": sid, "quote": text}],
                    _source_records=[{"id": sid, "role": "user", "message_id": sid, "text": text}],
                    relation_model=decision)

            old = await create("Cedar uses MySQL.\nCedar uses SQLite only after migration approval.",
                               fact={"value": "MySQL", "atomic": False})
            new = await create("Cedar now uses PostgreSQL.", fact={"value": "PostgreSQL"},
                               target=old["id"], fragment="Cedar uses MySQL.")
            remainder = await app.memory_service.get(old["id"], touch=False)
            check("automatic_replacement_uses_new_product_logic", usage_state(new) == "current" and
                  not new.get("user_confirmed") and bool(new.get("supersedes")))
            check("conditional_remainder_not_promoted", usage_state(remainder) == "qualified" and
                  not app.memory_service._eligible_for_context(remainder))
            body = (await app.tool_call("memory_read", {"memory_id": old["id"], "purpose": "all"},
                                        run_id=None, mode="safe"))["memory"]
            check("qualified_remainder_still_readable", body["usage_state"] == "qualified" and
                  "only after" in body["content"])

            proposal = await create("If we migrate next year, it would be because licensing is expensive.",
                                    fact={"kind": "plan", "value": "PostgreSQL"})
            views = [(await app.tool_call("memory_read", {"memory_id": m["id"], "purpose": "all"},
                                         run_id=None, mode="safe"))["memory"] for m in (proposal, new)]
            messages = [{"role": "tool", "name": "memory_read", "content": dumps({"memory": m})} for m in views]
            plan = quality.memory_verification_plan("I want to know why Cedar changed its database.", "Unknown",
                                                    messages, candidates=views)
            check("requester_does_not_hide_project", plan["anchor"]["id"] == new["id"] and "user" not in plan["subjects"])
            check("plan_cause_does_not_cover_completed_event", "reason" in plan["evidence_gaps"] and plan["needed"])
            candidates = await app.memory_service.relevant_items("Cedar database")
            check("native_initial_candidates_keep_limit_and_state", len(candidates) <= 5 and
                  old["id"] not in {m["id"] for m in candidates} and proposal["id"] not in {m["id"] for m in candidates})

            # More than five extracted facts must survive the actual adapter's
            # write path; five is still the initial retrieval limit.
            assertions = [f"My lasting option-{i} is token-{i}." for i in range(7)]
            source = " ".join(assertions)
            async def extract_batch(prompt):
                return dumps({"memories": [{"title": f"Option {i}", "summary": text, "content": text,
                    "confidence": .95, "fact": {"subject":"user", "topic":"configuration",
                    "attribute":f"option-{i}", "value":f"token-{i}", "kind":"current"},
                    "provenance":[{"source_id":"batch-user","quote":text}]} for i,text in enumerate(assertions)]})
            async def preserve_existing(prompt):
                payload = json.loads(prompt[prompt.index('{"incoming":'):])
                check("adjudication_exposes_target_edit_constraints", all("partial_edit_constraints" in m for m in payload['candidates']))
                return dumps({"relations":[{"id":m['id'],"relation":"coexist","confidence":.95,
                    "scope_match":"different", "incoming_quote":payload['incoming']['provenance'][0]['quote'],
                    "existing_quote":m['content'], "reason":"Distinct synthetic option attributes"} for m in payload['candidates']]})
            extracted = await app.memory_service.reflect(user_text=source, assistant_text="Noted", source_id="readiness-batch",
                source_records=[{"id":"batch-user","role":"user","text":source}], model=extract_batch,
                relation_model=preserve_existing)
            check("all_seven_extracted_items_survive_native_write", len(extracted)==7 and
                  {m['fact']['attribute'] for m in extracted}=={f'option-{i}' for i in range(7)})
            check("expanded_write_does_not_expand_initial_candidates",
                  len(await app.memory_service.relevant_items('lasting option')) <= 5)
            check("qualified_target_cannot_be_partially_edited", not partial_edit_constraints(proposal)['allowed'])

            await app.db.set_json("privacy_settings", {**privacy, "memory_review_mode": "review"})
            pending = await create("Cedar team prefers morning meetings.", fact={"attribute": "meeting preference"})
            hidden = (await app.tool_call("memory_read", {"memory_id": pending["id"], "purpose": "all"},
                                          run_id=None, mode="safe"))["memory"]
            check("manual_mode_still_blocks_unapproved_content", pending["status"] == "pending_review" and hidden.get("content_withheld"))
        finally:
            await close(app)
        check("no_network_attempts", not network_attempts)
    return checks
