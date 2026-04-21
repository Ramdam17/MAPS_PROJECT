"""MARL factorial setting abstraction (paper §B.4 Table 12, settings 1-6).

Per paper §B.4 preamble, student `train_meltingpot.py:112-142` maps the 6
factorial-like cells to ``(meta, cascade_iterations1, cascade_iterations2)``
tuples. We mirror that mapping in :class:`MarlSetting`.

References
----------
- Paper §2.2 + Table 12 (p. 30).
- Student ``external/paper_reference/marl_tmlr/onpolicy/scripts/train/train_meltingpot.py:112-142``.
- Port scope lock : ``docs/reviews/marl-scope-decisions.md`` §e.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

__all__ = ["MarlSetting"]


@dataclass(frozen=True)
class MarlSetting:
    """One entry of the 6-cell MARL factorial grid.

    Parameters
    ----------
    id : str
        Short slug used in config / output dir names. Canonical ids :
        ``"baseline"``, ``"cascade_1st_no_meta"``, ``"meta_no_cascade"``,
        ``"maps"``, ``"meta_cascade_2nd"``, ``"meta_cascade_both"``.
    label : str
        Human-readable label for logs / reports.
    meta : bool
        If True, use ``MAPSActor`` / ``MAPSCritic`` (with the second-order
        wager head). If False, use plain ``MAPPOActor`` / ``MAPPOCritic``.
    cascade_iterations1 : int
        Cascade depth on the 1st-order (actor / critic) RNN. ``1`` disables
        cascade (single forward pass, paper eq.6 degenerate). ``50`` = full
        paper cascade (student value ; paper §B.4 preamble says no cascade,
        but student code uses 50 for settings 2/4/6 — student code wins per
        Blindsight/AGL pattern).
    cascade_iterations2 : int
        Cascade depth on the 2nd-order (wager) path. Same semantics as above.
    """

    id: str
    label: str
    meta: bool
    cascade_iterations1: int
    cascade_iterations2: int

    @classmethod
    def from_dict(cls, d: DictConfig | dict) -> MarlSetting:
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", d["id"])),
            meta=bool(d["meta"]),
            cascade_iterations1=int(d["cascade_iterations1"]),
            cascade_iterations2=int(d["cascade_iterations2"]),
        )

    @property
    def has_cascade_1st(self) -> bool:
        return self.cascade_iterations1 > 1

    @property
    def has_cascade_2nd(self) -> bool:
        return self.cascade_iterations2 > 1
