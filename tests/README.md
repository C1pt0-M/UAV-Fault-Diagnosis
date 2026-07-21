# Test suites

The active Python test suite is discovered from the repository root by the
`[tool.pytest.ini_options]` configuration in `pyproject.toml`.

Run the same verification used by CI:

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\verify.py
```

Files matching `test_sim_factory_*.py` are excluded because the corresponding
`sim_factory` implementation was removed before the monorepo migration. They
remain ignored local history and are not part of the project test baseline.

Tests marked `external` validate BUAA/RflySim installations or ignored vendor
source that is not part of this repository. They are excluded from the active
baseline. Run them explicitly only in a prepared environment:

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pytest -m external -o addopts="" tests
```
