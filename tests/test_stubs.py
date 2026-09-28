"""Tests for the CLI contract: every subcommand exists, and the constants
and registries behind it are real, not placeholders.
"""

from click.testing import CliRunner

from hmda import __version__
from hmda.cli import main
from hmda.clean.sentinels import KNOWN_SENTINELS, UNCONFIRMED_SENTINELS
from hmda.fairness.mitigate import TECHNIQUE_REGISTRY
from hmda.model.features import EXCLUDED_COLUMNS


def test_version_is_importable_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_sentinel_map_contains_the_three_measured_sentinels():
    """HMDA codes missing values as NA, Exempt and 8888; measured 2026-09-11: all three confirmed present.
    1111 was CORRECTED 2026-09-13: national data (36,734,685-row `lar` table)
    shows income=1111 (236 rows) indistinguishable from its neighbours
    (1108-1114 range 236-334) with none of the clustering genuine round
    values show (1100->1209, 1200->3537); it is ordinary $1,111,000 income,
    not a sentinel, and is not in KNOWN_SENTINELS."""
    all_known = set().union(*KNOWN_SENTINELS.values())
    assert "NA" in all_known
    assert "Exempt" in all_known
    assert "8888" in all_known
    assert "1111" not in all_known


def test_mitigation_technique_registry_has_exactly_three_entries():
    """One mitigation technique per stage, no more."""
    assert len(TECHNIQUE_REGISTRY) == 3
    assert set(TECHNIQUE_REGISTRY) == {
        "reweighing",
        "fair_constrained_gbm",
        "per_group_threshold",
    }


def test_protected_class_columns_are_excluded_from_model_features():
    """M7: protected-class fields are excluded from the feature matrix."""
    for col in ("derived_race", "derived_ethnicity", "derived_sex"):
        assert col in EXCLUDED_COLUMNS


def test_cli_registers_every_subcommand():
    """hmda ingest | audit | model | mitigate | recourse | explain | report | verify."""
    expected = {
        "ingest",
        "audit",
        "model",
        "mitigate",
        "recourse",
        "explain",
        "report",
        "verify",
    }
    assert expected.issubset(set(main.commands.keys()))


def test_cli_subcommands_exit_nonzero_and_print_not_implemented():
    """`ingest` and `verify --counts` are implemented now and no longer print
    "not implemented". `model` is still stubbed, so it is used here to check
    that a genuinely unimplemented subcommand still exits nonzero with that
    message. `ingest`'s real behavior (argument validation, not a stub
    message) is covered by tests/test_ingest.py."""
    runner = CliRunner()
    result = runner.invoke(main, ["model"])
    assert result.exit_code != 0
    assert "not implemented" in result.output
