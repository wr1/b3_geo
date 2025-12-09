from __future__ import annotations

from pydantic import BaseModel


class Planform(BaseModel):
    """Planform parameters along the blade span."""

    z: list[tuple[float, float]]  # (rel_span, z_value)
    chord: list[tuple[float, float]]
    thickness: list[tuple[float, float]]
    twist: list[tuple[float, float]]
    dx: list[tuple[float, float]]
    dy: list[tuple[float, float]]
    npchord: int = 200


class Airfoil(BaseModel):
    """Airfoil configuration."""

    path: str
    name: str
    thickness: float


class BladeConfig(BaseModel):
    """Overall blade configuration."""

    planform: Planform
    airfoils: list[Airfoil] = []
