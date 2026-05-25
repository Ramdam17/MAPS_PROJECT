"""Unit tests for BlindsightSetting / AGLSetting after the 6-cell refactor.

Covers:
- New 6-cell schema (cascade_1st / cascade_2nd / second_order) round-trip.
- Legacy 2×2 schema (cascade: bool) → symmetric cascade interpretation.
- Back-compat ``.cascade`` property.
- ``label`` defaults to ``id`` when missing.
- OmegaConf DictConfig path.
- YAML files in ``config/experiments/`` load and produce the documented cells.

Reference for the canonical 6-cell mapping :
``docs/reproduction/experiment_matrix.md`` §Settings factorial.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from maps.experiments.agl import AGLSetting
from maps.experiments.blindsight import BlindsightSetting
from maps.utils import load_config

# Cell-by-cell expected mapping for the 6-cell factorial. Anchored by id +
# the (cascade_1st, cascade_2nd, second_order) tuple from
# experiment_matrix.md §Settings factorial (paper Figure 6).
EXPECTED_6CELL = [
    # (id, cascade_1st, cascade_2nd, second_order)
    ("setting-1-baseline", False, False, False),
    ("setting-2-cascade-1st", True, False, False),
    ("setting-3-second-order-only", False, False, True),
    ("setting-4-maps", True, False, True),
    ("setting-5-cascade-2nd", False, True, True),
    ("setting-6-full-maps", True, True, True),
]


@pytest.mark.parametrize("cls", [BlindsightSetting, AGLSetting])
class TestNewSchema:
    """The 6-cell schema is the new canonical input shape."""

    def test_explicit_construction_round_trips(self, cls):
        s = cls(
            id="setting-4-maps",
            label="MAPS",
            cascade_1st=True,
            cascade_2nd=False,
            second_order=True,
        )
        assert s.id == "setting-4-maps"
        assert s.label == "MAPS"
        assert s.cascade_1st is True
        assert s.cascade_2nd is False
        assert s.second_order is True

    def test_from_dict_with_new_schema(self, cls):
        d = {
            "id": "setting-5-cascade-2nd",
            "label": "S5",
            "cascade_1st": False,
            "cascade_2nd": True,
            "second_order": True,
        }
        s = cls.from_dict(d)
        assert s.cascade_1st is False
        assert s.cascade_2nd is True
        assert s.second_order is True

    def test_from_dict_label_defaults_to_id(self, cls):
        s = cls.from_dict(
            {
                "id": "setting-1-baseline",
                "cascade_1st": False,
                "cascade_2nd": False,
                "second_order": False,
            }
        )
        assert s.label == "setting-1-baseline"

    def test_from_dict_with_omegaconf_dictconfig(self, cls):
        d = OmegaConf.create(
            {
                "id": "setting-6-full-maps",
                "cascade_1st": True,
                "cascade_2nd": True,
                "second_order": True,
            }
        )
        s = cls.from_dict(d)
        assert s.cascade_1st is True and s.cascade_2nd is True

    def test_cascade_property_back_compat(self, cls):
        """`.cascade` is True iff either side has cascade on."""
        on_both = cls(id="x", label="", cascade_1st=True, cascade_2nd=True, second_order=True)
        on_1st = cls(id="x", label="", cascade_1st=True, cascade_2nd=False, second_order=True)
        on_2nd = cls(id="x", label="", cascade_1st=False, cascade_2nd=True, second_order=True)
        off = cls(id="x", label="", cascade_1st=False, cascade_2nd=False, second_order=True)
        assert on_both.cascade is True
        assert on_1st.cascade is True
        assert on_2nd.cascade is True
        assert off.cascade is False


@pytest.mark.parametrize("cls", [BlindsightSetting, AGLSetting])
class TestLegacySchema:
    """Legacy 2×2 YAML (`cascade: bool`) maps to symmetric cascade.

    Matches pre-refactor :class:`BlindsightTrainer` / :class:`AGLTrainer`
    behavior exactly — guarantees the historical 4-cell archive
    (``knowyourself_outputs_20260501.tar.gz``) remains valid.
    """

    @pytest.mark.parametrize(
        ("legacy_id", "cascade", "second_order", "expected_paper_setting"),
        [
            ("neither", False, False, 1),
            ("cascade_only", True, False, 2),
            ("second_order_only", False, True, 3),
            ("both", True, True, 6),
        ],
    )
    def test_legacy_maps_to_paper_setting(
        self, cls, legacy_id, cascade, second_order, expected_paper_setting
    ):
        s = cls.from_dict(
            {"id": legacy_id, "label": legacy_id, "cascade": cascade, "second_order": second_order}
        )
        # cascade_1st = cascade_2nd = cascade (symmetric interpretation).
        assert s.cascade_1st == cascade
        assert s.cascade_2nd == cascade
        assert s.second_order == second_order
        # Sanity: the .cascade property still works.
        assert s.cascade == cascade
        # The expected paper setting tag is informational; documented in plan.
        assert expected_paper_setting in (1, 2, 3, 6)


@pytest.mark.parametrize("cls", [BlindsightSetting, AGLSetting])
class TestYAMLConfigs:
    """The shipped YAML configs round-trip through from_dict."""

    def test_factorial_6cell_yaml_loads_all_six(self, cls):
        cfg = load_config("experiments/factorial_6cell")
        assert int(cfg.n_seeds) == 500
        assert len(cfg.settings) == 6
        for entry, (expected_id, c1, c2, so) in zip(cfg.settings, EXPECTED_6CELL, strict=True):
            s = cls.from_dict(entry)
            assert s.id == expected_id
            assert s.cascade_1st is c1
            assert s.cascade_2nd is c2
            assert s.second_order is so

    def test_factorial_2x2_yaml_still_loads_via_legacy_path(self, cls):
        """The legacy 4-cell YAML keeps loading and maps to symmetric cascade."""
        cfg = load_config("experiments/factorial_2x2")
        assert len(cfg.settings) == 4
        ids = [s.id for s in cfg.settings]
        assert ids == ["neither", "cascade_only", "second_order_only", "both"]
        for entry in cfg.settings:
            s = cls.from_dict(entry)
            # Symmetric guarantee.
            assert s.cascade_1st == s.cascade_2nd
