from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class pchip_curve(BaseModel):
    """Planform curve interpolated with PCHIP."""

    type: Literal["pchip"] = "pchip"
    points: list[tuple[float, float]]


class bezier_curve(BaseModel):
    """Planform curve as piecewise cubic Bezier with Inkscape-style direction handles.

    handles[i] = (in_dx, in_dy, out_dx, out_dy) — offsets from points[i].
    If empty, default C1 handles are computed from PCHIP derivatives.
    """

    type: Literal["bezier"] = "bezier"
    points: list[tuple[float, float]]
    handles: list[tuple[float, float, float, float]] = []


planform_curve = Annotated[
    Union[pchip_curve, bezier_curve], Field(discriminator="type")
]

_PLANFORM_FIELDS = ("z", "chord", "thickness", "twist", "dx", "dy")


class Planform(BaseModel):
    """Planform parameters along the blade span."""

    z: planform_curve
    chord: planform_curve
    thickness: planform_curve
    twist: planform_curve
    dx: planform_curve
    dy: planform_curve
    npchord: int = 200

    @model_validator(mode="before")
    @classmethod
    def coerce_plain_lists(cls, data):
        """Auto-coerce plain [[x, y], ...] lists to pchip_curve — preserves backward compat."""
        for field in _PLANFORM_FIELDS:
            val = data.get(field)
            if isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
                data[field] = {"type": "pchip", "points": val}
        return data


class Airfoil(BaseModel):
    """Airfoil configuration."""

    path: str = ""
    name: str
    thickness: float
    coordinates: list[list[float]] | None = None  # inline [[x, y], ...] replaces path


class BladeConfig(BaseModel):
    """Overall blade configuration."""

    planform: Planform
    airfoils: list[Airfoil] = []
