# b3_geo

3D blade geometry evaluator: planform + airfoil stack → the **W vtk** — one
closed polyline per span section, carrying the UACS / SDACS / GBCS coordinate
frames and section fields every downstream package reads.

- Entry point: `build_w_vtk(planform, airfoil_stack, ...)`
- Docs: `packages/b3_geo/` pages in the documentation site
  (`master_docs/src/content/docs/packages/b3_geo/`)
- Internal audit notes: `docs/b3_geo_audit.md`
