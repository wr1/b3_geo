"""Equivalence tests: b3_geo.realise() == b3_geo.Blade.get_sections() on same input.

b3_geo is imported ONLY in this test file — never at runtime in b3_geo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_blade.templates.blade_mini import BladeMini
from b3_geo import realise

FIXTURE = Path(__file__).parent / "fixtures" / "blade_mini.yml"


@pytest.mark.integration
def test_sections_match_legacy():
    """b3_geo sections must match b3_geo sections on the same blade definition."""
    from b3_geo.models import BladeConfig  # test-only import
    from b3_geo.core.blade import Blade as LegacyBlade  # test-only import

    blade = BladeMini()
    spans = np.linspace(0.02, 0.98, 20)

    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(), spans)

    with FIXTURE.open() as f:
        yml = yaml.safe_load(f)
    legacy_cfg = BladeConfig(**yml["geometry"])
    legacy_blade = LegacyBlade(legacy_cfg)
    legacy_sections = legacy_blade.get_sections(spans)

    np.testing.assert_allclose(rb.sections, legacy_sections, atol=1e-4,
                               err_msg="b3_geo sections diverge from b3_geo baseline")


def test_sections_shape_consistent():
    """Smoke: realise() produces sections with expected shape without b3_geo."""
    blade = BladeMini()
    spans = np.linspace(0.0, 1.0, 15)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(), spans, npchord=50)
    assert rb.sections.shape == (15, 50, 3)
    assert rb.spans_abs.shape == (15,)


def test_fields_schema():
    """Fields DataFrame has all required b3_drp2 columns."""
    blade = BladeMini()
    spans = np.linspace(0.0, 1.0, 10)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(), spans, npchord=20)
    required = {"s", "z", "chord_frac", "panel_id", "dist_from_le", "dist_from_te"}
    assert required.issubset(rb.fields.columns)


def test_web_chord_frac_accessible():
    blade = BladeMini()
    spans = np.linspace(0.0, 1.0, 10)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(), spans)
    cf = rb.web_chord_frac("main_spar")
    assert cf.shape == (10,)
