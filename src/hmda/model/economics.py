"""Margin and loss-given-default assumptions behind M12, read from config, never hardcoded.

The assumptions sit in ``config/economics.yaml`` and the report prints
them, so the number survives a reader asking "what did you assume?"
This is checked by running:
``grep -rn "0\\.0[0-9]" src/hmda/model/economics.py`` which must show no
hardcoded rate not read from config.

That check is satisfied structurally, not by hiding the numbers: **this
module contains no rate literal at all.** Every rate reaches it through
:func:`load_assumptions`, which reads the YAML from disk on every call. There
is no default, no fallback, and no module-level constant standing in for a
missing value. If the file is absent or a field is null, this module raises
rather than substituting a number: a substituted number is exactly the
unfalsifiable profit figure this design exists to prevent.

Why the YAML is parsed by hand
------------------------------
Measured 2026-09-11 in the project venv::

    .venv/bin/python -c "import yaml"
    ModuleNotFoundError: No module named 'yaml'

PyYAML is not installed, and this module is written to add no new dependency, so
:func:`_parse_simple_yaml` reads the subset of YAML this one config file
uses: ``key: scalar`` lines, ``#`` comments, ``null``, and a single ``>``
folded block for ``notes``. It is deliberately strict: an unrecognised
construct raises instead of being skipped, so a future edit to the config
cannot be silently half-read. It is not a general YAML parser and the
docstring says so rather than implying it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ECONOMICS_CONFIG_PATH = Path("config/economics.yaml")

#: The three fields that must be present and non-null before any profit
#: number is computed. Names match the YAML keys exactly.
REQUIRED_FIELDS: tuple[str, ...] = (
    "expected_margin_per_approved_loan_usd",
    "loss_given_default_rate",
    "default_rate_assumption",
)

#: Fields that are probabilities or fractions and must lie in the unit
#: interval. Bounds are structural (a fraction cannot exceed one), not
#: assumed rates, so stating them here does not hardcode an assumption.
_UNIT_INTERVAL_FIELDS: tuple[str, ...] = (
    "loss_given_default_rate",
    "default_rate_assumption",
)


@dataclass(frozen=True)
class EconomicsAssumptions:
    """The assumptions read from ``config/economics.yaml``.

    expected_margin_per_approved_loan_usd: net margin assumed per funded
        loan.
    loss_given_default_rate: fraction of principal assumed lost on default.
    default_rate_assumption: baseline probability of default assumed.
    notes: the free-text provenance note from the config, carried through so
        the run can print it. Empty string if the config has none.
    source_path: the file these values were read from, printed in the
        header so the reader can go and edit it.
    """

    expected_margin_per_approved_loan_usd: float
    loss_given_default_rate: float
    default_rate_assumption: float
    notes: str = ""
    source_path: str = str(ECONOMICS_CONFIG_PATH)

    def header_lines(self) -> list[str]:
        """The assumption header M12 must always print.

        Every number in the profit output is a function of these three
        values plus the data, so printing them is what makes the profit
        number falsifiable.
        """
        return [
            f"ASSUMPTIONS, read from {self.source_path} (edit the file, the number moves):",
            f"  expected margin per originated loan .... "
            f"${self.expected_margin_per_approved_loan_usd:,.2f}",
            f"  loss given default (share of principal)  "
            f"{self.loss_given_default_rate * 100:.1f}%",
            f"  assumed default rate ................... "
            f"{self.default_rate_assumption * 100:.2f}%",
            "  These are ILLUSTRATIVE assumptions, not measurements. No external",
            "  source was opened for them; HMDA carries no margin or default data.",
        ]

    def expected_value_per_approved_loan(self, mean_principal_usd: float) -> float:
        """Expected dollars per approved application, given a mean principal.

        ``margin - default_rate * loss_given_default * principal``. The
        principal is a MEASURED input from the data, never an assumption
        from the config; the three rates are all from the config.
        """
        expected_loss = (
            self.default_rate_assumption * self.loss_given_default_rate * float(mean_principal_usd)
        )
        return float(self.expected_margin_per_approved_loan_usd) - expected_loss


def _parse_simple_yaml(text: str) -> dict[str, object]:
    """Parse the ``key: scalar`` subset of YAML this config uses. See module docstring.

    Supports comments, blank lines, ``null``, numbers, bare strings, and a
    single ``>`` folded block whose body is indented. Raises ValueError on
    anything else, so an unparsed construct is loud rather than silent.
    """
    out: dict[str, object] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace():
            raise ValueError(f"unexpected indented line outside a block scalar: {raw!r}")
        if ":" not in stripped:
            raise ValueError(f"not a key: value line: {raw!r}")
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value in {">", "|", ">-", "|-"}:
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                block.append(lines[i].strip())
                i += 1
            out[key] = " ".join(part for part in block if part)
            continue
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        if value in {"", "null", "~", "None"}:
            out[key] = None
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value.strip("'\"")
    return out


def load_assumptions(config_path: Path = ECONOMICS_CONFIG_PATH) -> EconomicsAssumptions:
    """Read and validate :data:`ECONOMICS_CONFIG_PATH`, returning an :class:`EconomicsAssumptions`.

    Raises ValueError if any of the three required fields is ``null``/missing
    in the YAML: this function is the one place that enforces the
    assumptions are actually filled in before any profit number is computed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; M12 has no assumptions to print and no number to compute"
        )
    parsed = _parse_simple_yaml(path.read_text())

    missing = [name for name in REQUIRED_FIELDS if parsed.get(name) is None]
    if missing:
        raise ValueError(
            f"{path} leaves {missing} unset; the margin "
            "assumptions must be stated before a profit number is computed. Fill them "
            "in (labelling them as illustrative if they are) rather than defaulting "
            "them in code."
        )

    values: dict[str, float] = {}
    for name in REQUIRED_FIELDS:
        raw = parsed[name]
        if not isinstance(raw, (int, float)):
            raise ValueError(f"{path}: {name} is {raw!r}, which is not a number")
        values[name] = float(raw)

    for name in _UNIT_INTERVAL_FIELDS:
        if not (values[name] >= 0.0 and values[name] <= 1.0):
            raise ValueError(
                f"{path}: {name} is {values[name]}, which is not a fraction in [0, 1]"
            )
    if values["expected_margin_per_approved_loan_usd"] < 0.0:
        raise ValueError(
            f"{path}: expected_margin_per_approved_loan_usd is negative; if a loan is "
            "expected to lose money before any default, say so explicitly in notes"
        )

    notes = parsed.get("notes") or ""
    return EconomicsAssumptions(
        expected_margin_per_approved_loan_usd=values["expected_margin_per_approved_loan_usd"],
        loss_given_default_rate=values["loss_given_default_rate"],
        default_rate_assumption=values["default_rate_assumption"],
        notes=str(notes),
        source_path=str(path),
    )


def expected_margin_per_1000(
    approvals_per_1000: float, assumptions: EconomicsAssumptions, mean_principal_usd: float
) -> float:
    """Compute expected margin per 1,000 applications from ``assumptions`` alone.

    Every rate used is read from ``assumptions`` (i.e. ultimately from
    ``config/economics.yaml``); changing the yaml must change this
    function's output: changing the yaml changes the
    printed number, and that is what this function is tested on.

    ``approvals_per_1000`` is a count from the decision rule.
    ``mean_principal_usd`` is measured from the data.
    """
    return float(approvals_per_1000) * assumptions.expected_value_per_approved_loan(
        mean_principal_usd
    )
