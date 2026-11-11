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

## Score event replay

New or versioned research callers reuse the lightweight module directly:

```python
from uav_fault.online.event_replay import aggregate_scores

result = aggregate_scores(times_s, fault_probabilities, policy=declared_policy)
```

The input timeline contains already-produced scores. Unavailable scores remain
nonfinite gaps; `inference_every` does not resample this sequence. Results retain
`confirmations`, `finalized_events`, and `policy`. Replay observation times and
the aggregator's deployment timing metadata remain distinct.

`tests/test_event_replay.py` compares every output field against the original
`aggregate_scores` function in `scripts/research/replay_m1_continuous.py`, whose
SHA256 is checked against the frozen M3 protocol. The reference is extracted
only inside tests, without executing research-script imports. Resetting the
emitting-host clock for each run permits exact comparisons of all metadata;
no fields or numeric differences are ignored. Synthetic sequences cover the
declared GBT source policy, the A1 policy, gaps, confirmation, duration filters,
merge delays and flush. An isolated process blocks ML and research imports.

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pytest -q tests/test_event_replay.py
```

Frozen entrypoints and historical audit code deliberately continue using their
original implementation. This addition does not change GBT online inference,
deployment configuration, thresholds, model assets, or prior experiment results.
