# b3_geo pre-refactor audit

## Public API (`__init__.py`)

```python
from b3_geo.realise import RealisedBlade, realise
from b3_geo.web_resolve import WebGeom
```

Exports: `realise`, `RealisedBlade`, `WebGeom`

## Rotation / scaling / translation / column-swap locations

| File | Lines | Operation |
|------|-------|-----------|
| `_section.py` | 43 | `pts[:, 1] -= 0.5` — chord centering |
| `_section.py` | 44 | `pts[:, :2] *= chords[i]` — scale by chord |
| `_section.py` | 45–49 | 2D rotation matrix applied to cols 0,1 by `-twist_deg * π/180` |
| `_section.py` | 50–52 | translate: add `(dx, dy, z)` |
| `utils/interpolation.py` | 14 | `data[:, [1, 0]]` — column swap on load |
| `utils/interpolation.py` | 20 | `data[:, [1, 0]]` — same swap in `airfoil_from_coordinates` |
| `web_resolve.py` | 74–78 | `np.cos/sin(twists_rad)` → chord/normal direction vectors |

All of the above must consolidate into `transforms.py` in the new design.

## Section / Blade producers

| File | Function | Returns |
|------|----------|---------|
| `_section.py` | `build_sections()` | `np.ndarray (N_spans, npchord, 3)` — GBCS section coords |
| `realise.py` | `realise()` | `RealisedBlade` — sections + webs + fields + hints |
| `utils/interpolation.py` | `build_sections_poly()` | `pv.StructuredGrid` — blade surface grid |

All to be replaced by `wire.py` / `build.py` producing W vtk.

## Dependencies

| Package | Files that import it |
|---------|---------------------|
| numpy | `_section.py`, `realise.py`, `web_resolve.py`, `fields.py`, `hint_resolve.py`, `utils/interpolation.py` |
| scipy | `_section.py` (`interp1d`), `utils/interpolation.py` (`CubicSpline`, `PchipInterpolator`) |
| pandas | `realise.py`, `fields.py` |
| pyvista | `utils/interpolation.py` (lazy import in `build_sections_poly`) |

pyvista is **not** listed in `pyproject.toml` — must be added.
pandas is a dependency of the legacy `fields.py` only — not needed in new design.
scipy remains needed for airfoil interpolation in `wire.py`.
