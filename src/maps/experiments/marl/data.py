"""MARL shared dataclasses (E.7 scaffold).

Helper dataclasses for rollout buffers, per-agent batches, etc. Most of the
runtime ``SharedReplayBuffer`` / ``SeparatedReplayBuffer`` logic lives inside
student's ``onpolicy/utils/*_buffer.py`` files — we will port the minimum
needed for PPO (not a full rllib-compat buffer).

E.9 scope — not implemented here.
"""

from __future__ import annotations

__all__: list[str] = []

# E.9 will add buffer dataclasses (RolloutBuffer, AgentBatch, etc.)
