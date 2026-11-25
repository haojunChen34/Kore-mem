"""Isolated MemProbe evaluation utilities. No personal application startup."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
ROOT = Path(os.environ.get("KORE_AGENT_ROOT", str(PROJECT_ROOT.parent / "Kore-Agent"))).expanduser().resolve()
if not (ROOT / "backend/kore_next/application.py").is_file():
    raise RuntimeError("Set KORE_AGENT_ROOT to a Kore-Agent checkout containing backend/kore_next.")
# Product imports must not load a daily application's .env or data directories.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
CONFIG_PATH = Path(os.environ.get("KORE_MEM_CONFIG", str(PROJECT_ROOT / "configs/memprobe.example.json"))).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ISOLATED = PROJECT_ROOT / ".local" / "config"
for key, value in {"KORENEXT_DATA_DIR": ISOLATED / "data", "KORENEXT_WORKSPACE_DIR": ISOLATED / "workspace",
                   "KORENEXT_USER_SKILLS_DIR": ISOLATED / "skills"}.items():
    os.environ[key] = str(value)
for key, value in CONFIG["runtime_settings"].items():
    if key not in {"max_steps", "run_warning_tokens", "run_finalization_tokens", "run_hard_tokens",
                   "run_warning_model_calls", "run_finalization_model_calls", "run_hard_model_calls"}:
        raise ValueError("Unsupported runtime setting: " + key)
    os.environ["KORENEXT_" + key.upper()] = str(value)
VENDOR = HERE / "vendor"
INITIAL_RUN = HERE / "runs" / "2026-11-19-initial"
RUN_NAME = os.environ.get("KORE_MEMPROBE_RUN", "local-run")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", RUN_NAME) or RUN_NAME == INITIAL_RUN.name:
    raise ValueError("Use a new simple run name; the initial run is immutable")
RUN = HERE / "runs" / RUN_NAME
EXPECTED_MODEL = CONFIG["model"]["model"]
REVISION = "9068d8d9a2a47268fe6f7601b99a56a283c37be5"
sys.path.insert(0, str(ROOT / "backend"))


def utcnow():
    return datetime.now(UTC).isoformat()


def dump(path, value):
    path = Path(path).resolve()
    if path.is_relative_to(INITIAL_RUN.resolve()):
        raise ValueError("Refusing to modify the initial benchmark run")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def product_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "backend" / "kore_next").rglob("*.py"))}


def credentials():
    """Use only the benchmark config and its named environment secret."""
    from kore_next.config import Settings
    public = dict(CONFIG["model"])
    if "api_key" in public:
        raise ValueError("Keep API keys in the named environment variable, not JSON.")
    key = os.environ.get(CONFIG.get("api_key_env", "KORE_MEM_API_KEY"), "")
    profile = {**public, "api_key": key}
    saved = {**CONFIG["roles"], "default_provider": public["provider"],
             "providers": [public], "_benchmark_memory_policy": CONFIG.get("memory_policy", {})}
    personal = Settings(api_key="", tavily_api_key="", provider=public["provider"],
                        model=public["model"], base_url=public["base_url"])
    return personal, saved, {public["provider"]: profile}


def effective_models(saved, profiles):
    main = profiles[saved["default_provider"]]
    result = {"answer": public_profile(main)}
    for role in ("reflection", "maintenance", "summary"):
        spec = saved.get(role, {"enabled": True, "inherit_main": True})
        if not spec.get("enabled", True):
            if role == "summary":
                result[role] = {"enabled": False}
                continue
            raise ValueError("Required memory role disabled: " + role)
        profile = main if spec.get("inherit_main", True) else {
            **profiles[spec["provider"]], "model": spec.get("model") or profiles[spec["provider"]]["model"]}
        result[role] = public_profile(profile)
    result["relationship"] = dict(result["reflection"])
    return result


def validate_models(saved, profiles):
    models = effective_models(saved, profiles)
    for role, config in models.items():
        if config.get("enabled") is False:
            continue
        if config.get("model") != EXPECTED_MODEL:
            raise ValueError(f"{role} must use {EXPECTED_MODEL}; found {config.get('model')}")
        if os.environ.get("KORE_MEMPROBE_ALLOW_LIVE") == "1" and not profiles[config["provider"]].get("api_key"):
            raise ValueError("Credential missing for " + role)
    return models


@contextmanager
def offline_network_guard():
    """No TCP connections during offline checks; MockTransport remains usable."""
    import socket
    from unittest.mock import patch
    original = socket.socket.connect
    attempts = []

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("blocked_tcp_connection")
            raise RuntimeError("Network disabled for offline MemProbe validation")
        return original(sock, address)

    with patch.object(socket.socket, "connect", connect), patch.object(socket.socket, "connect_ex", connect):
        yield attempts


def public_profile(profile):
    return {k: v for k, v in profile.items() if k in
            ("provider", "model", "base_url", "api_style", "context_window", "reasoning_effort", "temperature")}


def redact(message, profiles):
    text = str(message)
    for p in profiles.values():
        if p.get("api_key"):
            text = text.replace(p["api_key"], "[REDACTED]")
    return text


def network_setup():
    current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    os.environ["NO_PROXY"] = ",".join(filter(None, [current, "localhost", "127.0.0.1"]))


def load_episodes():
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if sha256(VENDOR / relative) != expected:
            raise RuntimeError("Pinned vendor file changed: " + relative)
    return [json.loads(s) for s in (VENDOR / "data/suite56/episodes.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]


def replay_units(episode):
    """Native post-run inputs: current user text and last assistant text per turn.

    Retain every raw message and its role. Tool outputs and intermediate assistant
    text are NOT injected into fields where the product does not receive them.
    A user turn without an assistant final reply cannot trigger native post-run.
    Session boundaries are preserved; no synthetic assistant responses are made.
    No gold, session types, phase labels, or narrative metadata cross this boundary.
    """
    units, orphans = [], []
    for si, session in enumerate(episode["sessions"]):
        current = None
        for mi, msg in enumerate(session["dialogue"]):
            role, content = msg["role"], msg["content"]
            if role not in ("user", "assistant", "tool_output") or not isinstance(content, str):
                raise ValueError("Unsupported input role or nontext content")
            entry = {"session": si, "message": mi, "role": role, "content": content}
            if role == "user":
                if current is not None:
                    units.append(current)
                current = {"session": si, "user_message_index": mi, "user": content,
                           "assistant": "", "assistant_message_index": None, "raw": [entry]}
            elif current is None:
                orphans.append(entry)
            else:
                current["raw"].append(entry)
                if role == "assistant":
                    current["assistant"] = content
                    current["assistant_message_index"] = mi
        if current is not None:
            units.append(current)
    for index, unit in enumerate(units):
        unit["unit_id"] = f"u{index:04d}"
    return units, orphans
