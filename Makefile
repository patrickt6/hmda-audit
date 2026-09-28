PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: test bench reproduce report

test:
	$(PYTEST) -q

bench:
	$(PY) bench/compare_pandas.py

reproduce:
	$(PY) -m hmda.cli ingest --year 2023
	$(PY) -m hmda.cli audit --all-years
	$(PY) -m hmda.cli report

report:
	$(PY) -m hmda.cli report
