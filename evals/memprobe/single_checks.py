"""Offline tests of native orchestration, reporting and historical comparison."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx

import execution
import runner
from common import dump, load_episodes, offline_network_guard, product_hashes, sha256
from execution import counts, pilot_ids, read_plan
from runtime import BenchApp, BenchMemory, OFFLINE_MOCK, install_hooks
from check_memory_search import isolated_settings, close
from aggregate_results import compare_rows, aggregate_results
from timeout_policy import ObservedHTTPClient


async def run_checks(checks=None):
    checks = [] if checks is None else checks
    def check(name, value, detail=None):
        checks.append({"name": "single_" + name, "passed": bool(value), "detail": detail})
        if not value:
            raise AssertionError(name + ": " + str(detail))
    episodes = load_episodes()
    check("224_planned_answers", counts(episodes, ["native"]) == {"episodes_planned":56,"questions_planned":224,"answers_planned":224})
    pilot = pilot_ids(episodes)
    check("four_fixed_pilot_scenes", len(pilot) == 4 and len({e["paradigm"] for e in episodes if e["episode_id"] in pilot}) == 4)
    check("pilot_order_independent", pilot == pilot_ids(list(reversed(episodes))))
    model = {"provider":"openai", "model":"gpt-5.6-sol", "base_url":"https://offline.invalid/v1",
             "api_key":"offline-fixture", "api_style":"chat_completions", "context_window":1000000}
    saved = {"default_provider":"openai", "providers":[model],
             "reflection":{"enabled":True,"inherit_main":True}, "maintenance":{"enabled":True,"inherit_main":True}}
    profiles = {"openai":model}
    wire = []
    def handler(request):
        body = json.loads(request.content)
        wire.append(body)
        prompt = body["messages"][0]["content"]
        if prompt.startswith("Extract only stable"):
            sources = json.loads(prompt[prompt.index('{"sources":'):])["sources"]
            user = next(s for s in sources if s["role"] == "user" and not s.get("context_only"))
            content = json.dumps({"memories":[{"title":"Cedar database", "summary":"Cedar uses Quartz",
                "content":"Project Cedar uses Quartz as its database.","confidence":.94,"reason":"Explicit durable decision",
                "fact":{"subject":"user","topic":"Cedar","attribute":"database","value":"Quartz","kind":"current","atomic":True},
                "provenance":[{"source_id":user["id"],"quote":user["text"]}]}],"core_updates":{}})
            if user["text"] == "Malformed response fixture":
                content = '{"memories":['
        else:
            content = '{"answer":"Quartz"}'
        usage = {"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}
        if body.get("stream"):
            chunk = {"choices":[{"index":0,"delta":{"content":content}}],"usage":usage}
            return httpx.Response(200,text="data: "+json.dumps(chunk)+"\n\ndata: [DONE]\n\n",headers={"content-type":"text/event-stream"})
        return httpx.Response(200,json={"choices":[{"message":{"role":"assistant","content":content}}],"usage":usage})
    def client(*args,**kwargs):
        return ObservedHTTPClient(*args,**kwargs,transport=httpx.MockTransport(handler))
    install_hooks()
    mock = OFFLINE_MOCK.set(True)
    before = product_hashes()
    try:
        with offline_network_guard() as attempts, tempfile.TemporaryDirectory(prefix="memprobe-native-") as folder:
            root = Path(folder)
            plan = {"mode":"single","arms":["native"],"candidate_source":"product_runtime"}
            dump(root/"execution_plan.json",plan)
            check("native_mode",read_plan(root)["arms"] == ["native"])
            dump(root/"legacy/formal_config.json",{})
            check("legacy_mode_preserved",read_plan(root/"legacy")["arms"] == ["A","B"])
            cases = [{"episode_id":f"SYNTH-{i}","group_id":f"GROUP-{i}","paradigm":"interference","latent_spec":{},
                      "sessions":[{"dialogue":[{"role":"user","content":"Please remember: Project Cedar uses Quartz as its database."},
                                                   {"role":"assistant","content":"Noted."}]}],
                      "probes":[{"question":q,"probe_type":"current_value","gold_answer":"Quartz"}
                                for q in ("What database does Project Cedar use?","Which database is used by Cedar?")]} for i in range(2)]
            cases[0]["sessions"][0]["dialogue"][:0] = [
                {"role":"user","content":"Malformed response fixture"},
                {"role":"assistant","content":"Noted."}]
            with patch.object(runner,"RUN",root),patch.object(execution,"RUN",root),patch.object(httpx,"AsyncClient",client):
                controller = runner.Controller(cases,isolated_settings(root/"unused"),saved,profiles,2)
                await controller.batch(cases)
                check("controller_completes",not controller.state["blockers"] and all(s["status"] == "complete" for s in controller.state["episodes"].values()),controller.state["blockers"])
                failed_turn = json.loads((root/"episodes/SYNTH-0/ingestion/turns/u0000/result.json").read_text(encoding="utf-8"))
                check("controller_continues_after_recorded_product_failure", failed_turn["status"] == "product_failure"
                      and controller.state["episodes"]["SYNTH-0"]["post_run_failures"] == ["u0000"]
                      and controller.state["episodes"]["SYNTH-0"]["ingested"] == 2)
                check("one_answer_per_question",len(list(root.glob("episodes/*/answers/q*/native/score.json"))) == 4)
                check("no_AB_directories",not list(root.glob("episodes/*/answers/q*/A")) and not list(root.glob("episodes/*/answers/q*/B")))
                results = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("episodes/*/answers/q*/native/trial/result.json")]
                check("runtime_candidates_present",all(r["candidates"] and r["candidate_source"] == "product_runtime" for r in results))
                check("correct_native_scoring",all(json.loads(p.read_text(encoding="utf-8"))["match"] == "correct" for p in root.glob("episodes/*/answers/q*/native/score.json")))
                for ep in cases:
                    d = root/"episodes"/ep["episode_id"]
                    meta = json.loads((d/"snapshot.json").read_text(encoding="utf-8"))
                    rr = [r for r in results if r["episode_id"] == ep["episode_id"]]
                    check(ep["episode_id"]+"_isolated_starts",all(r["starting_memory_fingerprint"] == meta["memory_fingerprint"] for r in rr))
                    check(ep["episode_id"]+"_snapshot_immutable",sha256(d/"snapshot.db") == meta["sha256"])
                original_calls=len(wire)
                await controller.batch(cases)
                check("resume_does_not_repeat_calls",len(wire)==original_calls)
                dump(root/"data_audit.json",{"totals":{"user_turns":2}})
                estimate=controller.estimate([e["episode_id"] for e in cases])
                check("estimate_uses_native_task_count",estimate["formal_answers_planned"] == 4 and estimate["pilot_answers"] == 4)
                preserved_state = copy.deepcopy(controller.state)
                dump(root/"execution_scope.json", {"episode_ids":["SYNTH-0"], "pause_on_product_failure":True})
                from execution import scoped_episodes
                check("scope_filters_explicit_ids", [e["episode_id"] for e in scoped_episodes(cases)] == ["SYNTH-0"])
                controller.state["episodes"]["SYNTH-0"]["status"] = "queued"
                controller.persist()
                paused = runner.Controller(cases,isolated_settings(root/"unused"),saved,profiles,1)
                await paused.episode(cases[0])
                check("product_failure_pauses_before_next_turn", paused.state["status"] == "PAUSED"
                      and paused.state["blockers"][0]["kind"] == "product_failure_review"
                      and paused.state["episodes"]["SYNTH-0"]["ingested"] == 1 and len(wire) == original_calls)
                (root/"execution_scope.json").unlink()
                dump(root/"execution_plan.json", {**plan,"pause_on_product_failure":True})
                controller.state = copy.deepcopy(preserved_state)
                controller.state["episodes"]["SYNTH-0"]["status"] = "queued"
                controller.persist()
                full_paused = runner.Controller(cases,isolated_settings(root/"unused"),saved,profiles,1)
                await full_paused.episode(cases[0])
                check("full_scope_product_failure_pauses", full_paused.state["status"] == "PAUSED"
                      and full_paused.state["blockers"][0]["kind"] == "product_failure_review"
                      and len(wire) == original_calls)
                failure_path = root/"episodes/SYNTH-0/ingestion/turns/u0000/result.json"
                review = {"action":"retain_failure_and_continue", "reason":"Malformed fixture; preserve omission",
                          "reviewed_at":"2026-01-01T00:00:00Z", "result_sha256":sha256(failure_path)}
                dump(root/"reviewed_failures.json", {"failures":{"SYNTH-0/u0000":review}})
                check("review_accepts_exact_failed_evidence", runner.reviewed_post_run_failure("SYNTH-0","u0000",failed_turn))
                check("review_rejects_changed_result", not runner.reviewed_post_run_failure("SYNTH-0","u0000",{**failed_turn,"error":"different"}))
                check("review_rejects_other_unit", not runner.reviewed_post_run_failure("SYNTH-0","u0001",failed_turn))
                dump(root/"reviewed_failures.json", {"failures":{"SYNTH-0/u0000":{**review,"result_sha256":"wrong"}}})
                check("review_rejects_wrong_hash", not runner.reviewed_post_run_failure("SYNTH-0","u0000",failed_turn))
                dump(root/"reviewed_failures.json", {"failures":{"SYNTH-0/u0000":review}})
                controller.state = copy.deepcopy(preserved_state)
                controller.state["episodes"]["SYNTH-0"]["status"] = "queued"
                controller.persist()
                resumed = runner.Controller(cases,isolated_settings(root/"unused"),saved,profiles,1)
                await resumed.episode(cases[0])
                check("reviewed_failure_resumes_without_retry", not resumed.state["blockers"]
                      and resumed.state["episodes"]["SYNTH-0"]["status"] == "complete" and len(wire) == original_calls)
                check("reviewed_failure_preserves_original_result", sha256(failure_path) == review["result_sha256"])
                dump(root/"execution_plan.json",plan)
                controller.state = preserved_state
                controller.state["arms"]=["A","B"]
                controller.persist()
                try:
                    runner.Controller(cases,isolated_settings(root/"unused"),saved,profiles,2)
                except RuntimeError:
                    check("mixed_mode_resume_rejected",True)
                else:
                    check("mixed_mode_resume_rejected",False)
            app=BenchApp(isolated_settings(root/"guard"),saved,profiles,frozen=True,arm="native")
            try:
                await app.initialize_benchmark()
                for attr,method in (("fixed_candidates","relevant_items"),("fixed_notices","context_notices")):
                    setattr(app,attr,[])
                    try:
                        await getattr(app.memory_service,method)("Cedar")
                    except RuntimeError:
                        check(attr+"_override_rejected",True)
                    else:
                        check(attr+"_override_rejected",False)
                    setattr(app,attr,None)
                check("answer_learning_frozen",(await app.db.get_json("privacy_settings"))["auto_memory_enabled"] is False)
            finally:
                await close(app)
            try:
                aggregate_results(root,cases,check_baseline=False)
            except RuntimeError as exc:
                check("mock_formal_report_rejected","Mock calls" in str(exc))
            else:
                check("mock_formal_report_rejected",False)
            # Exercise numeric aggregation/audits with synthetic disk fixtures only.
            # First verify the mock-rejection gate above; now explicitly convert
            # this temporary fixture ledger to test the non-mock aggregation path.
            assert root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
            for p in root.glob("episodes/*/**/model_calls/*.json"):
                c=json.loads(p.read_text(encoding="utf-8"))
                c.update(offline_mock_transport=False, synthetic_test_fixture=True)
                dump(p,c)
            controller.state.update(status="COMPLETE",arms=["native"])
            dump(root/"progress.json",controller.state)
            dump(root/"formal_config.json",{"evaluation":plan,"product_sources":before,
                 "adapter_sources":{name:sha256(Path(__file__).parent/name) for name in runner.FROZEN_FILES}})
            dump(root/"product_baseline.json",{"product_sources":before})
            dump(root/"question_manifest.json",[{"episode_id":e["episode_id"],"probe_index":i} for e in cases for i in range(len(e["probes"]))])
            m=aggregate_results(root,cases,check_baseline=False)
            baseline=json.loads((root/"reports/per_question.json").read_text(encoding="utf-8"))
            m=aggregate_results(root,cases,baseline_rows=baseline,check_baseline=False)
            check("full_native_fixture_report",m["complete"] and m["summary"]["correct"]==4)
            check("all_call_costs_once",m["all_formal_usage"]["requests"]==7 and m["all_formal_usage"]["input_tokens"]==700)
            check("answer_and_build_costs_separate",m["usage_by_phase"]["answer"]["input_tokens"]==400 and m["usage_by_phase"]["extraction"]["input_tokens"]==300)
            check("failed_extraction_reported_once",len(m["memory_processing_failures"])==1
                  and m["memory_processing_failures"][0]["episode_id"]=="SYNTH-0"
                  and m["memory_processing_failures"][0]["unit_id"]=="u0000"
                  and m["memory_processing_failures"][0]["error"]=="Invalid memory extraction result")
            check("version_comparison_complete",m["historical_comparison"]["paired_questions"]==4 and m["historical_comparison"]["net_correct"]==0)
            config = json.loads((root/"formal_config.json").read_text(encoding="utf-8"))
            scoped_config = {**config,"execution_subset":{"episode_ids":["SYNTH-0"]}}
            dump(root/"formal_config.json",scoped_config)
            controller.state["status"]="PILOT_COMPLETE"
            dump(root/"progress.json",{**controller.state,"episodes":{"SYNTH-0":controller.state["episodes"]["SYNTH-0"]}})
            subset_report=aggregate_results(root,cases,baseline_rows=baseline,check_baseline=False)
            check("subset_report_correct_denominator",subset_report["complete"] and subset_report["summary"]["planned"]==2
                  and subset_report["historical_comparison"]["paired_questions"]==2)
            dump(root/"formal_config.json",config)
            controller.state["status"]="COMPLETE"
            dump(root/"progress.json",controller.state)
            aggregate_results(root,cases,baseline_rows=baseline,check_baseline=False)
            import verify, final_audit
            with patch.object(execution,"RUN",root),patch.object(verify,"RUN",root),patch.object(verify,"load_episodes",return_value=cases):
                check("native_integrity_audit",verify.main()==0)
                # Check all three turns, including the preserved extraction failure.
                with patch.object(final_audit,"RUN",root),patch.object(final_audit,"load_episodes",return_value=cases):
                    check("native_final_audit",final_audit.main()==0)
            source_candidate=root/"episodes/SYNTH-0/candidates/q0.json"
            changed=json.loads(source_candidate.read_text(encoding="utf-8"));changed["source"]="fixed_shared_candidates"
            dump(source_candidate,changed)
            with patch.object(execution,"RUN",root),patch.object(verify,"RUN",root),patch.object(verify,"load_episodes",return_value=cases):
                check("integrity_detects_non_native_candidate_source",verify.main()!=0)
            # Dedicated aggregation fixtures never leave the temporary test directory.
            reports=root/"report_fixture"
            dump(reports/"execution_plan.json",plan)
            fixture=[{"episode_id":"REPORT","group_id":"REPORT-GROUP","paradigm":"interference","sessions":[],
                      "probes":[{"question":f"Synthetic question {i}","gold_answer":"Quartz","probe_type":t}
                                for i,t in enumerate(("current_value","previous_value","change_detection","conflict_value","source","temporal"))]}]
            labels=("correct","partial","incorrect","execution_error")
            for i,label in enumerate(labels):
                path=reports/"episodes/REPORT/answers"/f"q{i}"/"native"
                status="product_failure" if label=="execution_error" else "completed"
                dump(path/"trial/result.json",{"status":status,"answer":"synthetic fixture","calls":[],"seconds":1})
                dump(path/"score.json",{"episode_id":"REPORT","probe_index":i,"arm":"native","probe_type":fixture[0]["probes"][i]["probe_type"],"match":label,"execution_status":status})
            dump(reports/"progress.json",{"status":"COMPLETE","episodes":{"REPORT":{"status":"complete"}},"blockers":[]})
            m=aggregate_results(reports,fixture,check_baseline=False)
            check("partial_report_not_complete",not m["complete"] and m["summary"]["evaluated"]==4)
            check("full_denominator_retained",m["summary"]["correct"]==1 and m["summary"]["planned"]==6 and m["summary"]["accuracy"]==1/6)
            check("partial_not_counted_correct",m["summary"]["partial"]==1 and m["summary"]["incorrect"]==1)
            check("errors_and_unexecuted_separated",m["summary"]["execution_error"]==1 and m["summary"]["not_scored"]==2)
            check("no_usage_not_free",m["summary"]["complete_usage_tasks"]==0 and m["summary"]["mean_input_tokens_complete_usage"] is None)
            check("all_six_categories_present",len(m["summary"]["by_type"])==6)
            check("native_composite_metrics_present",all(k in m["summary"]["native"] for k in ("plasticity","stability","source_fidelity","split","non_split","temporal_fidelity")))
            check("aggregation_creates_no_markdown", not list(reports.rglob("*.md")))
            documents = (reports/"README.md", reports/"TOTAL_TEST_REPORT.md", reports/"RESUME_NOTES.md")
            for document in documents:
                document.write_text("User-maintained document.\n", encoding="utf-8")
            document_bytes = {document: document.read_bytes() for document in documents}
            aggregate_results(reports,fixture,check_baseline=False)
            check("aggregation_preserves_existing_documents",
                  all(document.read_bytes() == content for document,content in document_bytes.items()))
            manifest=json.loads((reports/"reports/delivery_manifest.json").read_text(encoding="utf-8"))
            check("evidence_manifest_excludes_documents",
                  not any(name.endswith(".md") for name in manifest["files"]))
            new=json.loads((reports/"reports/per_question.json").read_text(encoding="utf-8"))
            old=copy.deepcopy(new)
            old[0].update(match="incorrect",correct=False)
            old[1].update(match="correct",correct=True)
            cmp=compare_rows(new,old)
            check("both_improvement_and_regression",cmp["paired_questions"]==4 and cmp["improved"]==1 and cmp["regressed"]==1 and cmp["net_correct"]==0)
            check("cost_missing_excluded_not_imputed",cmp["usage_complete_pairs"]==0 and cmp["input_change"] is None)
            old[0]["gold"]="changed"
            try:
                compare_rows(new,old)
            except RuntimeError:
                check("different_gold_comparison_rejected",True)
            else:
                check("different_gold_comparison_rejected",False)
            try:
                compare_rows(new,[*new,new[0]])
            except RuntimeError:
                check("duplicate_baseline_rejected",True)
            else:
                check("duplicate_baseline_rejected",False)
            # A successful retry must not erase the cost of a failed attempt.
            active=reports/"episodes/REPORT/answers/q0/native/trial"
            call={"id":"final-call","phase":"answer","episode_id":"REPORT","unit_id":"q0",
                  "arm":"native","status":"completed","usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110}}
            final_result={"status":"completed","answer":"synthetic fixture","calls":[call],"seconds":1}
            dump(active/"result.json",final_result)
            aggregate_results(reports,fixture,check_baseline=False)
            prior=json.loads((reports/"reports/per_question.json").read_text(encoding="utf-8"))
            check("result_only_cost_fallback",prior[0]["usage"]["input_tokens"]==100)
            dump(active/"model_calls/final-call.json",call)
            archive=reports/"episodes/REPORT/environment_attempts/q0-native-attempt-1/trial"
            successful={**call,"id":"archived-call","usage":{"input_tokens":200,"output_tokens":20,"total_tokens":220}}
            failed={**call,"id":"failed-call","status":"error","usage":None}
            dump(archive/"result.json",{"status":"environment_error","answer":None,
                 "calls":[successful,failed],"seconds":3,"tool_observations":[{"name":"tool_catalog"}]})
            dump(archive/"model_calls/archived-call.json",successful)
            dump(archive/"model_calls/failed-call.json",failed)
            m=aggregate_results(reports,fixture,baseline_rows=prior,check_baseline=False)
            row=json.loads((reports/"reports/per_question.json").read_text(encoding="utf-8"))[0]
            check("archived_cost_included_once",row["usage"]["requests"]==3 and row["usage"]["input_tokens"]==300
                  and row["usage"]["output_tokens"]==30 and m["all_formal_usage"]["input_tokens"]==300)
            check("archived_unknown_cost_excludes_pair",row["usage"]["missing_usage_requests"]==1
                  and m["historical_comparison"]["usage_complete_pairs"]==0)
            check("archived_cost_does_not_change_grade",row["correct"] and m["summary"]["correct"]==1
                  and m["historical_comparison"]["net_correct"]==0)
            check("archived_time_and_tools_retained",row["seconds"]==4 and row["final_attempt_seconds"]==1
                  and row["archived_attempt_seconds"]==3 and row["archived_attempt_count"]==1
                  and row["tool_calls"]==[{"name":"tool_catalog"}])
            check("archived_evidence_linked",row["evidence"]["archived_results"]==[
                  "episodes/REPORT/environment_attempts/q0-native-attempt-1/trial/result.json"])
            dump(reports/"episodes/REPORT/answers/q0/A/result.json",{})
            try:
                aggregate_results(reports,fixture,check_baseline=False)
            except RuntimeError:
                check("paired_artifacts_rejected",True)
            else:
                check("paired_artifacts_rejected",False)
            check("network_blocked",not attempts)
    finally:
        OFFLINE_MOCK.reset(mock)
    check("product_sources_unchanged",product_hashes()==before)
    return checks


if __name__ == "__main__":
    import asyncio
    result=asyncio.run(run_checks())
    print(json.dumps({"status":"passed","checks":len(result)},ensure_ascii=False))
