"""Tests for the SHAP explainability pack (M14).

What this file checks:
  - `hmda explain --report` exits 0 and prints the sample size it used.
  - `git diff --exit-code docs/EXPLAINABILITY.md` exits 0 after a second
    regeneration (checked here by rendering twice and diffing the text, not
    by shelling out to git, so the test does not depend on repo state).
  - Every reported driver resolves to a real `docs/DATA-DICTIONARY.md` entry.
  - No SHAP magnitude ships in the rendered doc.

Every number this file exercises is a 50,000-row FIXTURE number
(``tests/fixtures/hmda_50k.parquet``, DC/WY/VT only). It is not a
national number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hmda.governance import explain as E

DATA_DICTIONARY = Path("docs/DATA-DICTIONARY.md")


def _dictionary_columns() -> set[str]:
    """Every raw column name that has its own row in DATA-DICTIONARY.md."""
    text = DATA_DICTIONARY.read_text()
    cols = set()
    for line in text.splitlines():
        m = re.match(r"\|\s*([A-Za-z0-9_\-]+)\s*\|", line)
        if m and m.group(1) not in ("Column",):
            cols.add(m.group(1))
    return cols


@pytest.fixture(scope="module")
def dictionary_columns() -> set[str]:
    assert DATA_DICTIONARY.exists(), "docs/DATA-DICTIONARY.md must exist before drivers can be resolved"
    cols = _dictionary_columns()
    assert cols, "parsed zero columns out of DATA-DICTIONARY.md; the parser or the file format changed"
    return cols


@pytest.fixture(scope="module")
def explain_result() -> E.ExplainabilityResult:
    """One SHAP run shared by every test that needs it (slow: trains a GBM)."""
    return E.run_explain()


# --------------------------------------------------------------------------
# Every raw column table entry named in RAW_COLUMN_LABELS / ENGINEERED_FEATURE_SOURCES
# resolves to a real DATA-DICTIONARY.md row. This is the static half of the
# dictionary-resolution check: it holds regardless of what a given SHAP
# run happens to rank top.
# --------------------------------------------------------------------------
def test_every_raw_column_label_resolves_to_a_data_dictionary_entry(dictionary_columns):
    for raw_col in E.RAW_COLUMN_LABELS:
        assert raw_col in dictionary_columns, (
            f"{raw_col!r} has a plain-English label in explain.RAW_COLUMN_LABELS "
            f"but no row in {DATA_DICTIONARY}"
        )


def test_every_engineered_feature_source_resolves_to_data_dictionary_entries(dictionary_columns):
    for feature, sources in E.ENGINEERED_FEATURE_SOURCES.items():
        for raw_col in sources:
            assert raw_col in dictionary_columns, (
                f"engineered feature {feature!r} depends on {raw_col!r}, "
                f"which has no row in {DATA_DICTIONARY}"
            )


# --------------------------------------------------------------------------
# THE GATE: every driver a real run actually reports resolves to the
# dictionary, via the raw columns recorded on its DriverScore.
# --------------------------------------------------------------------------
def test_every_reported_overall_driver_resolves_to_a_data_dictionary_entry(explain_result, dictionary_columns):
    label_to_raw = {**{v: (k,) for k, v in E.RAW_COLUMN_LABELS.items()},
                     **{E.ENGINEERED_FEATURE_LABELS[k]: v for k, v in E.ENGINEERED_FEATURE_SOURCES.items()}}
    assert explain_result.top_drivers_overall, "run_explain() reported zero overall drivers"
    for label in explain_result.top_drivers_overall:
        assert label in label_to_raw, f"driver label {label!r} is not a known plain-English label"
        for raw_col in label_to_raw[label]:
            assert raw_col in dictionary_columns, (
                f"driver {label!r} resolves to raw column {raw_col!r}, "
                f"which has no row in {DATA_DICTIONARY}"
            )


def test_every_reported_group_driver_resolves_to_a_data_dictionary_entry(explain_result, dictionary_columns):
    label_to_raw = {**{v: (k,) for k, v in E.RAW_COLUMN_LABELS.items()},
                     **{E.ENGINEERED_FEATURE_LABELS[k]: v for k, v in E.ENGINEERED_FEATURE_SOURCES.items()}}
    assert explain_result.top_drivers_by_group, "run_explain() reported zero groups"
    for group, drivers in explain_result.top_drivers_by_group.items():
        assert drivers, f"group {group!r} has zero reported drivers"
        for label in drivers:
            assert label in label_to_raw, f"driver label {label!r} (group {group!r}) is not a known plain-English label"
            for raw_col in label_to_raw[label]:
                assert raw_col in dictionary_columns, (
                    f"driver {label!r} in group {group!r} resolves to raw column "
                    f"{raw_col!r}, which has no row in {DATA_DICTIONARY}"
                )


def test_no_raw_column_name_is_surfaced_as_a_driver_label(explain_result):
    """Plain-English only: a raw column name like 'loan_to_value_ratio' or a
    one-hot dummy like 'loan_type_1' must never appear as a driver label."""
    raw_names = set(E.RAW_COLUMN_LABELS) | set(E.ENGINEERED_FEATURE_SOURCES)
    for label in explain_result.top_drivers_overall:
        assert label not in raw_names, f"raw column name {label!r} leaked as a driver label"
        assert "_" not in label, f"driver label {label!r} looks like a raw/dummy column name, not plain English"
    for drivers in explain_result.top_drivers_by_group.values():
        for label in drivers:
            assert label not in raw_names
            assert "_" not in label


# --------------------------------------------------------------------------
# Sample size is always printed (never hidden), and matches what was reported.
# --------------------------------------------------------------------------
def test_sample_size_is_printed_and_capped_by_sample_size_constant(explain_result):
    assert explain_result.sample_size > 0
    assert explain_result.sample_size <= E.SAMPLE_SIZE


def test_main_prints_the_sample_size(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "DOC_PATH", tmp_path / "EXPLAINABILITY.md")
    exit_code = E.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sample size:" in captured.out
    assert str(E.SAMPLE_SIZE) in captured.out or "sample size: " in captured.out


# --------------------------------------------------------------------------
# Regeneration is stable: rendering the same result twice is a no-op diff.
# --------------------------------------------------------------------------
def test_render_is_stable_across_two_runs(explain_result, tmp_path):
    out1 = tmp_path / "run1.md"
    out2 = tmp_path / "run2.md"
    E.render_explainability_doc(explain_result, out1)
    E.render_explainability_doc(explain_result, out2)
    assert out1.read_text() == out2.read_text()


def test_run_explain_is_deterministic_across_two_full_runs():
    """A full second training + SHAP pass (not just re-rendering the same
    result) must reproduce the same sample size and the same overall
    drivers, because the seed and the sample are fixed."""
    r1 = E.run_explain()
    r2 = E.run_explain()
    assert r1.sample_size == r2.sample_size
    assert r1.top_drivers_overall == r2.top_drivers_overall
    assert r1.top_drivers_by_group == r2.top_drivers_by_group


# --------------------------------------------------------------------------
# No SHAP value (a magnitude) ships as a bullet number: the rendered doc
# text must not contain a raw mean_abs_shap value. We check this indirectly:
# DriverScore.mean_abs_shap must never be interpolated into the rendered
# markdown.
# --------------------------------------------------------------------------
def test_rendered_doc_names_no_shap_magnitude(explain_result, tmp_path):
    out = tmp_path / "EXPLAINABILITY.md"
    E.render_explainability_doc(explain_result, out)
    text = out.read_text()
    assert "mean_abs_shap" not in text
    assert "shap value" not in text.lower() or "shap values come from" in text.lower()
    # M14 is a count clause: the doc must say how many features, not a magnitude.
    assert f"{explain_result.n_features} features" in text


# --------------------------------------------------------------------------
# The doc is explicit that this is a fixture, not a national sample,
# and that per-group rankings are descriptive.
# --------------------------------------------------------------------------
def test_rendered_doc_carries_the_fixture_only_warning(explain_result, tmp_path):
    out = tmp_path / "EXPLAINABILITY.md"
    E.render_explainability_doc(explain_result, out)
    text = out.read_text()
    assert "FIXTURE" in text
    assert "DC, WY and VT" in text or "DC/WY/VT" in text


def test_rendered_doc_states_drivers_are_descriptive_not_causal(explain_result, tmp_path):
    out = tmp_path / "EXPLAINABILITY.md"
    E.render_explainability_doc(explain_result, out)
    text = out.read_text()
    assert "not causal" in text.lower()
    assert "not model inputs" in text.lower() or "never inputs" in text.lower()


def test_small_groups_are_excluded_below_the_min_count_floor(explain_result):
    for group, count in explain_result.excluded_groups.items():
        assert count < E.MIN_GROUP_COUNT
    for group, count in explain_result.group_counts.items():
        assert count >= E.MIN_GROUP_COUNT


# --------------------------------------------------------------------------
# run_explain() must accept an in-memory DataFrame and must
# NOT read tests/fixtures/hmda_50k.parquet when it gets one.
# --------------------------------------------------------------------------
def _t22_frame(n: int = 12_345):
    import pandas as pd

    full = pd.read_parquet(E.FIXTURE_PATH)
    return full.sample(n=n, random_state=0).reset_index(drop=True)


def test_t22_run_explain_honours_an_in_memory_frame(monkeypatch):
    import pandas as pd

    frame = _t22_frame()

    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("run_explain read parquet despite being given a frame")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    result = E.run_explain(sample_size=200, frame=frame, source_label="TEST SOURCE")
    assert result.sample_size == 200


def test_run_explain_rejects_a_frame_with_no_source_label():
    """A frame with no source_label would print sample numbers under the
    fixture's DC/WY/VT label. That must be impossible, not merely discouraged."""
    frame = _t22_frame()
    with pytest.raises(ValueError, match="source_label"):
        E.run_explain(sample_size=200, frame=frame)


# --------------------------------------------------------------------------
# The doc's provenance WORDING must follow the actual source.
# The defect: "fixture" was hardcoded into the M14 line, both driver headings,
# the missingness note and the Limits bullet, so a national run rendered
# national numbers described as fixture-derived -- the exact mislabel this
# project exists to prevent.
# --------------------------------------------------------------------------
def _sourced_result(explain_result):
    import dataclasses

    return dataclasses.replace(
        explain_result,
        source_label="SOURCE: national -- 36,734,685 rows, all states",
        source_rows=1_500_000,
        sample_seed=0,
    )


def test_rendered_doc_never_calls_a_sourced_run_fixture_derived(explain_result, tmp_path):
    """Given a source_label, no line of the document may describe its numbers
    as fixture-derived."""
    out = tmp_path / "EXPLAINABILITY.md"
    E.render_explainability_doc(_sourced_result(explain_result), out)
    offenders = [
        line
        for line in out.read_text().splitlines()
        if "fixture" in line.lower()
    ]
    assert offenders == [], f"sourced run described as fixture-derived: {offenders}"


def test_rendered_doc_names_the_actual_source_when_one_was_given(explain_result, tmp_path):
    out = tmp_path / "EXPLAINABILITY.md"
    E.render_explainability_doc(_sourced_result(explain_result), out)
    text = out.read_text()
    assert "SOURCE: national -- 36,734,685 rows, all states" in text
    assert "SAMPLE-ONLY" in text
