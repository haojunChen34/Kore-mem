"""Pinned rule scorer; gold never enters the runtime adapter."""
from __future__ import annotations

import sys

from common import VENDOR

sys.path.insert(0, str(VENDOR / "src"))
sys.path.insert(0, str(VENDOR / "src/eval"))
sys.path.insert(0, str(VENDOR / "src/analysis"))
import protocol
from compute_metrics import build_episode_meta, compute_baseline_metrics


def schema(probe_type):
    return protocol._PROBE_SCHEMAS[probe_type]


def grade(answer, probe, latent_spec):
    # Empty/error outputs are failures in the all-planned denominator, never a
    # correct abstention. Preserve the original scorer's label separately.
    if not isinstance(answer, str) or not answer.strip():
        return {"match": "execution_error", "reason": "No final answer"}
    return protocol.score_answer(answer, probe["gold_answer"], probe["probe_type"], spec=latent_spec)
