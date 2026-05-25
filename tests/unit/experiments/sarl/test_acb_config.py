"""Unit tests for :class:`maps.experiments.sarl.ACBConfig`.

Pinned by the 2026-05-19 freeway-validation regression : the original
``for_game`` used ``kwargs.setdefault("validation_every_episodes", 15)``
which had no effect because ``run_sarl.py`` always passes
``validation_every_episodes`` from the YAML (= 500). The freeway override
must run unconditionally.
"""

from __future__ import annotations

from pathlib import Path

from maps.experiments.sarl.actor_critic import ACBConfig


def test_for_game_freeway_overrides_validation_cadence() -> None:
    """Freeway must validate every 15 episodes even when 500 is requested.

    Freeway has very long episodes (one game = ~1000 env steps), so with
    `num_frames=500_000` the run produces ~200 episodes. A 500-episode
    cadence triggers ZERO validation points. The reference (Young & Tian
    2019) explicitly drops it to 15 for this game.
    """
    cfg = ACBConfig.for_game(
        "freeway", seed=42, num_frames=500_000, output_dir=Path("/tmp/x"),
        validation_every_episodes=500,
    )
    assert cfg.validation_every_episodes == 15, (
        "Freeway must override validation_every_episodes to 15 "
        "(regression: 2026-05-19 production run hit n_validation_points=0 "
        "on all 3 freeway seeds because setdefault was masked)."
    )


def test_for_game_non_freeway_respects_caller() -> None:
    """Non-freeway games keep whatever the caller passed."""
    for game in ("space_invaders", "breakout", "seaquest", "asterix"):
        cfg = ACBConfig.for_game(
            game, seed=42, num_frames=500_000, output_dir=Path("/tmp/x"),
            validation_every_episodes=500,
        )
        assert cfg.validation_every_episodes == 500, (
            f"{game}: must not override validation_every_episodes (got {cfg.validation_every_episodes})"
        )


def test_for_game_freeway_default_when_not_passed() -> None:
    """Freeway gets 15 even without an explicit kwarg."""
    cfg = ACBConfig.for_game(
        "freeway", seed=42, num_frames=500_000, output_dir=Path("/tmp/x"),
    )
    assert cfg.validation_every_episodes == 15


def test_for_game_passes_paper_hyperparams() -> None:
    """Algorithmic constants reach the dataclass through for_game."""
    cfg = ACBConfig.for_game(
        "breakout", seed=42, num_frames=500_000, output_dir=Path("/tmp/x"),
        alpha=0.001, lambda_=0.9, gamma=0.95,
    )
    assert cfg.alpha == 0.001
    assert cfg.lambda_ == 0.9
    assert cfg.gamma == 0.95
