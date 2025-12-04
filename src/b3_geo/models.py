from pydantic import BaseModel
from typing import List, Tuple


class Planform(BaseModel):
    """Planform parameters along the blade span."""

    z: List[Tuple[float, float]]  # (rel_span, z_value)
    chord: List[Tuple[float, float]]
    thickness: List[Tuple[float, float]]
    twist: List[Tuple[float, float]]
    dx: List[Tuple[float, float]]
    dy: List[Tuple[float, float]]
    npchord: int = 200


class Airfoil(BaseModel):
    """Airfoil configuration."""

    path: str
    name: str
    thickness: float


class BladeConfig(BaseModel):
    """Overall blade configuration."""

    planform: Planform
    airfoils: List[Airfoil] = []
