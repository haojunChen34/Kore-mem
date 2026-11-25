"""Pin public benchmark artifacts and audit replay mappings without model calls."""
from __future__ import annotations

import collections
import concurrent.futures
import json
import subprocess

import httpx

from common import (ROOT, RUN, VENDOR, REVISION, credentials, dump,
                    load_episodes, product_hashes, public_profile, replay_units, sha256, utcnow, validate_models)
from execution import new_plan, pilot_ids

FILES = ['LICENSE', 'DATA_LICENSE', 'data/suite56/episodes.jsonl', 'data/suite56/latent_specs.json', 'src/eval/protocol.py', 'src/analysis/compute_metrics.py', 'src/paths.py', 'src/llm.py']


def fetch(relative):
    destination = VENDOR / relative
    if destination.exists():
        return relative, sha256(destination)
    url = f"https://raw.githubusercontent.com/jq-ding/MemProbe/{REVISION}/{relative}"
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=40, headers={"User-Agent": "Kore-MemProbe-evaluation"})
            r.raise_for_status()
            r.content.decode("utf-8")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(r.content)
            return relative, sha256(destination)
        except Exception:
            if attempt == 2:
                raise


def main():
    personal, saved, profiles = credentials()
    models = validate_models(saved, profiles)
    if (RUN / "formal_config.json").exists() or (RUN / "episodes").exists():
        raise RuntimeError("A run already started; use its frozen configuration or a new run name")
    RUN.mkdir(parents=True, exist_ok=True)
    plan = new_plan()
    dump(RUN / "execution_plan.json", plan)
    hashes, errors = {}, []
    # Existing pinned files must be verified, never silently re-baselined.
    if (VENDOR / "manifest.json").exists():
        load_episodes()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fetch, p): p for p in FILES}
            for future in concurrent.futures.as_completed(futures):
                try:
                    name, value = future.result()
                    hashes[name] = value
                except Exception as exc:
                    errors.append({"path": futures[future], "error": str(exc)})
    if errors:
        dump(RUN / "download_errors.json", errors)
        raise RuntimeError("Public artifact download incomplete; see download_errors.json")
    if hashes:
        dump(VENDOR / "manifest.json", {"repository": "https://github.com/jq-ding/MemProbe",
             "revision": REVISION, "downloaded_at": utcnow(), "files": hashes})
    es = load_episodes()
    from kore_next.memory import MemoryService
    roles = collections.Counter(m["role"] for e in es for s in e["sessions"] for m in s["dialogue"])
    per_episode = []
    for e in es:
        units, orphans = replay_units(e)
        mapping = {"episode_id": e["episode_id"], "units": units, "orphan_messages": orphans}
        dump(RUN / "replay_audit" / (e["episode_id"] + ".json"), mapping)
        per_episode.append({"episode_id": e["episode_id"], "group_id": e["group_id"],
            "paradigm": e["paradigm"], "sessions": len(e["sessions"]), "questions": len(e["probes"]),
            "user_turns": len(units), "unanswered_user_turns": sum(not u["assistant"] for u in units),
            "native_reflection_triggers": sum(bool(u["assistant"]) and MemoryService.should_reflect(u["user"], u["assistant"]) for u in units),
            "tool_messages_not_direct_reflection_inputs": sum(m["role"] == "tool_output" for u in units for m in u["raw"]),
            "orphan_messages": len(orphans)})
    assert len(es) == 56 and sum(len(e["probes"]) for e in es) == 224
    baseline = {"created_at": utcnow(), "product_sources": product_hashes(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "profiles": {k: public_profile(v) for k, v in profiles.items()},
                "effective_models": models,
                "memory_policy": saved.get("_benchmark_memory_policy", {}),
                "saved_roles": {k: saved.get(k) for k in ("default_provider", "reflection", "maintenance", "summary", "run_budget")}}
    previous = RUN / "product_baseline.json"
    dump(previous, baseline)
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    (RUN / "initial_git_status.txt").write_text(git_status, encoding="utf-8")
    # Fixed groups cover all four paradigms, selected by IDs, before model results.
    pilot_groups = ["GRP-INT-01", "GRP-MIS-01", "GRP-CON-01", "GRP-REC-01"]
    pilot = pilot_ids(es)
    report = {"status": "OFFLINE_DATA_READY", "revision": REVISION, "episodes": len(es),
              "questions": 224, "formal_answers_planned": 224, "evaluation": plan,
              "sessions": sum(len(e["sessions"]) for e in es), "roles": dict(roles),
              "totals": {k: sum(e[k] for e in per_episode) for k in
                         ("user_turns", "unanswered_user_turns", "native_reflection_triggers", "tool_messages_not_direct_reflection_inputs", "orphan_messages")},
              "per_episode": per_episode, "pilot_selection": "First episode ID per paradigm, selected before execution", "pilot_episode_ids": pilot,
              "model_configuration": models, "api_calls": 0,
              "limitations": ["Native reflection uses the current turn and bounded interpretation context from the product.",
                              "Raw tool outputs are preserved for audit, never relabeled as user/assistant input.",
                              "No-final-response turns do not trigger post-run reflection; no replies are synthesized.",
                              "No official gold or semantic phase labels go to the model."],
              "timestamp": utcnow()}
    dump(RUN / "data_audit.json", report)
    dump(RUN / "question_manifest.json", [{"episode_id": e["episode_id"], "probe_index": i,
         "probe_type": p["probe_type"], "question": p["question"]} for e in es for i, p in enumerate(e["probes"])])
    print(json.dumps({k: report[k] for k in ("status", "episodes", "questions", "sessions", "roles", "totals", "pilot_episode_ids", "model_configuration")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
