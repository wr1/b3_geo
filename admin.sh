#!/bin/bash

ruff format
ruff check --fix > out.txt
git add src/b3_geo/api/loft.py
git commit src/b3_geo/api/loft.py -m 'summary of edits for src/b3_geo/api/loft.py'
git add src/b3_geo/utils/cache.py
git commit src/b3_geo/utils/cache.py -m 'summary of edits for src/b3_geo/utils/cache.py'
git add tests/__init__.py
git commit tests/__init__.py -m 'summary of edits for tests/__init__.py'
git add src/__init__.py
git commit src/__init__.py -m 'summary of edits for src/__init__.py'
git add src/b3_geo/utils/interpolation.py
git commit src/b3_geo/utils/interpolation.py -m 'summary of edits for src/b3_geo/utils/interpolation.py'
git add pyproject.toml
git commit pyproject.toml -m 'summary of edits for pyproject.toml'
git add src/b3_geo/cli/af.py
git commit src/b3_geo/cli/af.py -m 'summary of edits for src/b3_geo/cli/af.py'
git add tests/test_loft_step.py
git commit tests/test_loft_step.py -m 'summary of edits for tests/test_loft_step.py'
git add tests/test_examples.py
git commit tests/test_examples.py -m 'summary of edits for tests/test_examples.py'
git add tests/test_cli_clean.py
git commit tests/test_cli_clean.py -m 'summary of edits for tests/test_cli_clean.py'
git add src/b3_geo/api/loft_step.py
git commit src/b3_geo/api/loft_step.py -m 'summary of edits for src/b3_geo/api/loft_step.py'
git add src/b3_geo/api/af.py
git commit src/b3_geo/api/af.py -m 'summary of edits for src/b3_geo/api/af.py'
git add tests/test_cli_main.py
git commit tests/test_cli_main.py -m 'summary of edits for tests/test_cli_main.py'
git add tests/test_cli_af.py
git commit tests/test_cli_af.py -m 'summary of edits for tests/test_cli_af.py'
git add src/b3_geo/__init__.py
git commit src/b3_geo/__init__.py -m 'summary of edits for src/b3_geo/__init__.py'
git add tests/test_planform.py
git commit tests/test_planform.py -m 'summary of edits for tests/test_planform.py'
git add src/b3_geo/cli/planform.py
git commit src/b3_geo/cli/planform.py -m 'summary of edits for src/b3_geo/cli/planform.py'
git add tests/test_loft.py
git commit tests/test_loft.py -m 'summary of edits for tests/test_loft.py'
git add src/b3_geo/api/__init__.py
git commit src/b3_geo/api/__init__.py -m 'summary of edits for src/b3_geo/api/__init__.py'
git add tests/test_cli_planform.py
git commit tests/test_cli_planform.py -m 'summary of edits for tests/test_cli_planform.py'
git add examples/blend_naca.py
git commit examples/blend_naca.py -m 'summary of edits for examples/blend_naca.py'
git add tests/test_blade.py
git commit tests/test_blade.py -m 'summary of edits for tests/test_blade.py'
git add tests/test_af.py
git commit tests/test_af.py -m 'summary of edits for tests/test_af.py'
git add tests/test_interpolation.py
git commit tests/test_interpolation.py -m 'summary of edits for tests/test_interpolation.py'
git add src/b3_geo/cli/__init__.py
git commit src/b3_geo/cli/__init__.py -m 'summary of edits for src/b3_geo/cli/__init__.py'
git add examples/ex1.py
git commit examples/ex1.py -m 'summary of edits for examples/ex1.py'
git add tests/test_af_step.py
git commit tests/test_af_step.py -m 'summary of edits for tests/test_af_step.py'
git add src/b3_geo/api/planform.py
git commit src/b3_geo/api/planform.py -m 'summary of edits for src/b3_geo/api/planform.py'
git add tests/test_plotting.py
git commit tests/test_plotting.py -m 'summary of edits for tests/test_plotting.py'
git add tests/test_cli_loft.py
git commit tests/test_cli_loft.py -m 'summary of edits for tests/test_cli_loft.py'
git add README.md
git commit README.md -m 'summary of edits for README.md'
git add src/b3_geo/api/af_step.py
git commit src/b3_geo/api/af_step.py -m 'summary of edits for src/b3_geo/api/af_step.py'
git add src/b3_geo/cli/loft.py
git commit src/b3_geo/cli/loft.py -m 'summary of edits for src/b3_geo/cli/loft.py'
git add src/b3_geo/core/blade.py
git commit src/b3_geo/core/blade.py -m 'summary of edits for src/b3_geo/core/blade.py'
git add src/b3_geo/utils/__init__.py
git commit src/b3_geo/utils/__init__.py -m 'summary of edits for src/b3_geo/utils/__init__.py'
git add src/b3_geo/models.py
git commit src/b3_geo/models.py -m 'summary of edits for src/b3_geo/models.py'
git add src/b3_geo/utils/plotting.py
git commit src/b3_geo/utils/plotting.py -m 'summary of edits for src/b3_geo/utils/plotting.py'
git add tests/test_cache.py
git commit tests/test_cache.py -m 'summary of edits for tests/test_cache.py'
git add src/b3_geo/cli/clean.py
git commit src/b3_geo/cli/clean.py -m 'summary of edits for src/b3_geo/cli/clean.py'
uv run pytest -v >> out.txt