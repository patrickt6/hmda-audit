"""Generate docs/MODEL-CARD.md from the database, never hand-edited.

The card covers intended use, out-of-scope use, data, protected-class
handling, limits, and the explicit sentence that the model makes no credit
decisions. Two properties are checked: regeneration is stable
(``git diff --exit-code docs/MODEL-CARD.md`` after a re-run), and
``grep -c "not used to make credit decisions" docs/MODEL-CARD.md`` returns
1 or more.

HOW THIS MODULE KEEPS THE CARD HONEST
-------------------------------------
Every number the card prints is a :class:`Fact`. A ``Fact`` carries the
rendered value, the basis (full-file, sample-based with its size, registry,
or test suite) and a locator naming where the value was read from. Facts
come from three source-of-truth kinds, and from nowhere else:

* **the database** (``data/hmda.duckdb``) for full-file counts and shares,
  read with SQL at render time;
* **``docs/VERIFICATION.md``** ("Recorded model runs") for the sample-based
  model and mitigation runs (M7, M11), parsed by anchored regex
  so a reworded document fails loudly instead of drifting silently;
* **``docs/CONTROLS.md``**, parsed by :mod:`hmda.governance.controls`, for
  the control counts.

``verify_card`` then re-derives the facts and checks that EVERY numeric
token in the shipped card is a token of some fact value. A hand-edited
number is not a fact token, so the check fails. That is what
``hmda verify --card`` is for.

Nothing here re-runs a model. The national model jobs are recorded
measurements (see ``docs/VERIFICATION.md``, "Recorded model runs"); re-running them
costs hours and gigabytes and is not what a model card is for.

METRIC GRAMMAR
--------------
No abbreviation of a statistical score appears in the rendered card. A
ranking-quality number is written as "ranks denials better than a base-rate
guess", never as a three-letter score name. A sample-based number always
carries its sample size in the same sentence. No lender is named, by
policy.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Repository root: this file is ``<root>/src/hmda/governance/model_card.py``.
REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_CARD_PATH = Path("docs/MODEL-CARD.md")
RECORDED_RUNS_PATH = Path("docs/VERIFICATION.md")
CONTROLS_PATH = Path("docs/CONTROLS.md")
DUCKDB_PATH = Path("data/hmda.duckdb")

#: The literal sentence every rendered model card must contain.
NO_CREDIT_DECISIONS_SENTENCE = "This model is not used to make credit decisions."

#: Abbreviations the card may never contain (the metric grammar above).
#: Checked case-insensitively by :func:`banned_terms_in`.
BANNED_TERMS: tuple[str, ...] = ("SPD", "DIR", "EOD", "AUC", "R-squared", "Rsquared", "p-value")

_BANNED_RE = re.compile(r"SPD|DIR|EOD|AUC|R-?squared|p-value", re.IGNORECASE)

#: Every run of digits that a reader would read as a number. The lookbehind
#: drops identifiers such as ``M13`` and ``C-37`` and decimal continuations,
#: which are labels, not measurements.
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9.,\-])\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True)
class Fact:
    """One number in the card, with the basis and locator it came from.

    key: stable identifier used by the renderer and the verifier.
    value: the string exactly as it is rendered into the card.
    basis: how the number was produced, in the card's own vocabulary
        ("full-file, measured", "sample-based, n=1,500,000, seeds 0/1/2",
        "registry count", "test suite").
    source: a checkable locator -- a SQL statement against the database, or
        a file path plus the anchor text the value was parsed from.
    """

    key: str
    value: str
    basis: str
    source: str


def banned_terms_in(text: str) -> list[str]:
    """Return every banned abbreviation found in ``text`` (case-insensitive)."""
    return sorted({m.group(0) for m in _BANNED_RE.finditer(text)})


def number_tokens(text: str) -> list[str]:
    """Return every numeric token a reader would read as a number in ``text``."""
    return [m.group(0) for m in _NUMBER_TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------------
# Fact collection: the database
# --------------------------------------------------------------------------
def _db_facts(db_path: Path) -> list[Fact]:
    """Read the full-file counts and the Race Not Available share from DuckDB.

    One connection, three scalar aggregates over ``lar``. Measured at
    0.27s wall on the 36.7M-row file (2026-09-13), so this is cheap enough
    to run on every render, which is the point: the card cannot go stale
    against the data it describes.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        years = [
            int(y)
            for (y,) in con.execute(
                "SELECT DISTINCT CAST(activity_year AS INTEGER) AS yr FROM lar ORDER BY yr"
            ).fetchall()
        ]
        total_rows, rna_rows = con.execute(
            "SELECT count(*), "
            "count(*) FILTER (WHERE derived_race = 'Race Not Available') FROM lar"
        ).fetchone()
        lenders = int(con.execute("SELECT COUNT(DISTINCT lei) FROM lar").fetchone()[0])
    finally:
        con.close()

    rna_share = 0.0 if total_rows == 0 else rna_rows / total_rows
    db_locator = f"{db_path} table lar"
    return [
        Fact(
            "applications",
            f"{int(total_rows):,}",
            "full-file, measured",
            f"SELECT count(*) FROM lar -- {db_locator}",
        ),
        Fact(
            "lenders",
            f"{lenders:,}",
            "full-file, measured",
            f"SELECT COUNT(DISTINCT lei) FROM lar -- {db_locator}",
        ),
        Fact(
            "years",
            f"{len(years):,}",
            "full-file, measured",
            f"SELECT DISTINCT activity_year FROM lar -- {db_locator}",
        ),
        Fact(
            "first_year",
            f"{years[0]}" if years else "0",
            "full-file, measured",
            f"SELECT min(activity_year) FROM lar -- {db_locator}",
        ),
        Fact(
            "last_year",
            f"{years[-1]}" if years else "0",
            "full-file, measured",
            f"SELECT max(activity_year) FROM lar -- {db_locator}",
        ),
        Fact(
            "race_not_available_share",
            f"{rna_share:.2%}",
            "full-file, measured",
            "SELECT count(*) FILTER (WHERE derived_race = 'Race Not Available') / count(*) "
            f"FROM lar -- {db_locator}",
        ),
    ]


# --------------------------------------------------------------------------
# Fact collection: the recorded measurement document
# --------------------------------------------------------------------------
def _search(text: str, pattern: str, where: str):
    """Return the match groups of ``pattern`` in ``text``, or raise naming ``where``.

    Failing loudly is deliberate. If ``docs/VERIFICATION.md`` is
    reworded so an anchor no longer matches, the render must stop rather
    than fall back on a number baked into this file, because a number baked
    into this file is exactly the hand-edit ``hmda verify --card`` exists to catch.
    """
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(
            f"model card: could not read {where} from {RECORDED_RUNS_PATH}: "
            f"pattern {pattern!r} did not match. The recorded-measurement document "
            "changed shape; fix the anchor, do not hard-code the number."
        )
    return match.groups()


def _recorded_facts(numbers_path: Path) -> list[Fact]:
    """Parse the recorded sample-based runs and the test count out of the measurement doc."""
    text = numbers_path.read_text(encoding="utf-8")
    doc = str(numbers_path)

    (sample_n, seeds) = _search(
        text,
        r"SAMPLE_BASED \(n=([\d,]+); seeds ([0-9, ]+)\)",
        "the M7 sample size and seeds",
    )
    seed_list = "/".join(s.strip() for s in seeds.split(","))
    sample_basis = f"sample-based, n={sample_n}, seeds {seed_list}"

    (logistic,) = _search(
        text,
        r"ranks denials \*\*(\d+)%\*\* better than the base-rate baseline",
        "the M7 logistic-regression figure",
    )
    (boosted,) = _search(
        text,
        r"gradient-boosted trees \*\*(\d+)%\*\* better",
        "the M7 gradient-boosted-trees figure",
    )
    (gap_low, gap_high) = _search(
        text,
        r"cuts the\s+approval-rate gap \*\*(\d+)-(\d+)%\*\*",
        "the M11 gap-reduction range",
    )
    cost_seeds = _search(
        text,
        r"\|\s*`per_group_threshold`\s*\|[^|]*\|\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\s*\|",
        "the M11 accuracy-cost figures",
    )
    costs = sorted(float(c) for c in cost_seeds)

    return [
        Fact("sample_n", sample_n, sample_basis, f"{doc} 'Recorded model runs', ranking-quality status line"),
        Fact(
            "ranking_logistic",
            f"{logistic}%",
            sample_basis,
            f"{doc} 'Recorded model runs', ranking quality",
        ),
        Fact(
            "ranking_boosted",
            f"{boosted}%",
            sample_basis,
            f"{doc} 'Recorded model runs', ranking quality",
        ),
        Fact("gap_cut_low", f"{gap_low}%", sample_basis, f"{doc} 'Recorded model runs', mitigation"),
        Fact("gap_cut_high", f"{gap_high}%", sample_basis, f"{doc} 'Recorded model runs', mitigation"),
        Fact(
            "accuracy_cost_low",
            f"{costs[0]:.2f}",
            sample_basis,
            f"{doc} 'Recorded model runs', mitigation table",
        ),
        Fact(
            "accuracy_cost_high",
            f"{costs[-1]:.2f}",
            sample_basis,
            f"{doc} 'Recorded model runs', mitigation table",
        ),
    ]


# --------------------------------------------------------------------------
# Fact collection: the control registry
# --------------------------------------------------------------------------
def _control_facts(controls_path: Path) -> list[Fact]:
    """Count documented controls and the subset backed by a test, from the registry itself."""
    from hmda.governance import controls as _controls

    rows = _controls.load_controls(controls_path)
    backed = [c for c in rows if c.implemented]
    return [
        Fact(
            "controls_documented",
            f"{len(rows):,}",
            "registry count",
            f"{controls_path} rows parsed by hmda.governance.controls.load_controls",
        ),
        Fact(
            "controls_backed",
            f"{len(backed):,}",
            "registry count",
            f"{controls_path} rows whose test column is not '{_controls.NOT_IMPLEMENTED}'",
        ),
    ]


# --------------------------------------------------------------------------
# Fact collection: the model's own inputs
# --------------------------------------------------------------------------
def _feature_facts() -> list[Fact]:
    """Count the model inputs and confirm no protected attribute is among them."""
    from hmda.model import features as _features

    return [
        Fact(
            "feature_columns",
            f"{len(_features.FEATURE_COLUMNS):,}",
            "code constant",
            "src/hmda/model/features.py FEATURE_COLUMNS",
        )
    ]


def collect_facts(
    db_path: Path = DUCKDB_PATH,
    numbers_path: Path = RECORDED_RUNS_PATH,
    controls_path: Path = CONTROLS_PATH,
) -> dict[str, Fact]:
    """Gather every number the card is allowed to print, keyed by :attr:`Fact.key`."""
    facts: list[Fact] = []
    facts += _db_facts(db_path)
    facts += _recorded_facts(numbers_path)
    facts += _control_facts(controls_path)
    facts += _feature_facts()
    return {f.key: f for f in facts}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_HEADER = """<!--
GENERATED FILE. Do not hand-edit.
Regenerate: hmda report --model-card
Check:      hmda verify --card   (fails if any number here is not a measured fact)
Generator:  src/hmda/governance/model_card.py
-->
"""


def _card_text(facts: dict[str, Fact]) -> str:
    """Build the card body from ``facts``. Every numeral here is a fact value."""
    v = {k: f.value for k, f in facts.items()}
    lines: list[str] = []
    a = lines.append

    a(_HEADER)
    a("# Model card: HMDA mortgage denial model")
    a("")
    a(
        "This card is generated from the audited database and from the recorded "
        "measurement runs. It is not written by hand. Every number below appears "
        "in the fact table at the end of the card with the basis it was measured on "
        "and a locator a reviewer can open."
    )
    a("")

    a("## What the model is")
    a("")
    a(
        "Two models over United States mortgage application records: a base-rate "
        "reference that scores every applicant with the same denial rate, and a "
        f"gradient-boosted challenger trained on {v['feature_columns']} loan and property "
        "attributes. The study question is whether denial outcomes differ across "
        "protected groups once loan and property attributes are held level."
    )
    a("")

    a("## Intended use")
    a("")
    a(
        "Offline fairness measurement and model-risk documentation over historical, "
        "public application records. The model exists to support an audit: to rank "
        "applications by denial risk so that outcome gaps between groups can be "
        "compared at a matched risk level, and to give a mitigation study something "
        "to mitigate."
    )
    a("")

    a("## Out-of-scope use")
    a("")
    a(f"{NO_CREDIT_DECISIONS_SENTENCE}")
    a("")
    a(
        "It was never deployed, never served a request, never scored a real applicant, "
        "and never influenced any lending outcome. It ran offline over records of "
        "applications that were already decided by their lenders years before this "
        "work began. It is not underwriting software, not a pricing tool, not a "
        "scoring service, and not evidence about any named institution."
    )
    a("")
    a(
        "It is also not a finding of discrimination. A measured gap between groups is "
        "a statistical disparity that warrants review. Proving discrimination requires "
        "evidence this data does not hold."
    )
    a("")

    a("## Data")
    a("")
    a(
        f"Public national mortgage application records: {v['applications']} applications "
        f"from {v['lenders']} lenders over {v['years']} years, {v['first_year']} to "
        f"{v['last_year']}. Counts are read from the database at render time, over every "
        "row, not from a sample."
    )
    a("")
    a(
        "Training and evaluation for the recorded model runs used a stratified sample of "
        f"{v['sample_n']} rows rather than the whole file, because the model stage is "
        "memory-bound where the aggregation stages are not. Every model number in this "
        "card is therefore a sample number and says so."
    )
    a("")

    a("## Protected-attribute handling")
    a("")
    a(
        "Race, ethnicity, sex and age are not model inputs. The feature matrix is built "
        "from an allowlist, so a protected column cannot become a feature by being "
        "forgotten, and the build raises if a protected column reaches the matrix. "
        "Protected attributes are used only on the measurement side, to group outcomes "
        "after the fact."
    )
    a("")
    a(
        "Excluding them does not make the model blind to them. Loan and property "
        "attributes carry information about who applied, so a gap can survive the "
        "exclusion. That is the reason the audit measures outcomes rather than trusting "
        "the input list."
    )
    a("")
    a(
        f"Race is unreported on {v['race_not_available_share']} of national applications. "
        "That share is large enough to move any group comparison, so it travels with "
        "every disparity number in this project."
    )
    a("")

    a("## How well it ranks")
    a("")
    a(
        "Against a base-rate guess, which is the honest floor, the logistic reference "
        f"ranks denials {v['ranking_logistic']} better than chance and the "
        f"gradient-boosted challenger {v['ranking_boosted']} better, on a sample of "
        f"{v['sample_n']} rows, repeated at three random seeds that agreed to the printed "
        "precision. Better than chance at ranking is not the same as accurate for an "
        "individual applicant, and this card makes no individual-level accuracy claim."
    )
    a("")

    a("## Mitigation, and what it costs")
    a("")
    a(
        "A mitigation study measured what closing the gap costs. Applying a separate "
        "approval cut-off per group cut the widest between-group approval-rate gap by "
        f"{v['gap_cut_low']} to {v['gap_cut_high']}, at a cost of "
        f"{v['accuracy_cost_low']} to {v['accuracy_cost_high']} accuracy points, on a "
        f"sample of {v['sample_n']} rows across three seeds. The range is the result. "
        "No single seed is the headline."
    )
    a("")
    a(
        "That technique applies a different approval cut-off by race, which is disparate "
        "treatment on its face in the United States. It is measured for completeness of "
        "the trade-off study and is not presented as approved practice."
    )
    a("")

    a("## Known limits")
    a("")
    a(
        "- **The recourse study is unmeasurable, not merely unmeasured.** It asked how "
        "much an applicant would have to change to cross the approval line, compared "
        "across groups. The per-group medians reorder between random seeds, so the "
        "comparison has no stable answer at this sample size. It is reported as a "
        "negative result and its medians are not quoted."
    )
    a(
        "- **The pair of groups defining the widest gap is unstable.** The gap metric is "
        "a maximum over all group pairs, and the pair realising that maximum changed "
        "between seeds. The magnitude is reportable; the identity of the two groups is "
        "not, and is named nowhere in this project as a result."
    )
    a(
        "- **Protected attributes are not model inputs**, so the model cannot be read as "
        "measuring a lender's intent, and the disparities measured here are outcome "
        "differences, not mechanisms."
    )
    a(
        "- **Model numbers are sample-based.** The counts and shares in the data section "
        "cover every row; the ranking and mitigation numbers do not, and are labelled "
        "with their sample size wherever they appear."
    )
    a(
        "- **The margin study is not trustworthy enough to quote.** It has one seed and "
        "no measured spread, and its sign reversed against the earlier fixture run. No "
        "money figure appears in this card."
    )
    a(
        "- **Speed and memory are measured, but not quoted here.** The comparison "
        "against a single-machine pandas baseline was run on the full national file "
        "(results/engineering.json, from bench/compare_pandas.py). "
        "It describes the pipeline, not the "
        "model, so this card does not repeat the figures."
    )
    a(
        "- **Race is unreported on a large minority of records** (see the data section). "
        "Group comparisons are conditional on who reported."
    )
    a(
        "- **No lender is named** anywhere in this project, by policy, and no result is "
        "attributed to an institution."
    )
    a("")

    a("## Governance")
    a("")
    a(
        f"{v['controls_documented']} controls are documented for this project, of which "
        f"{v['controls_backed']} are backed by an automated test. Both numbers travel "
        "together: the first counts what is written down, the second counts what is "
        # The repo's test count is deliberately NOT stated here. It changes on
        # every commit, so a model card citing it is stale the moment it renders:
        # this sentence read 238 while the live suite was already at 255. A test
        # count is a property of the repository, not of the model, and belongs in
        # the README. The control counts above ARE model-risk properties and stay.
        "checked."
    )
    a("")

    a("## Fact table: every number in this card")
    a("")
    a("| number | value | basis | source |")
    a("| --- | --- | --- | --- |")
    for key in facts:
        f = facts[key]
        a(f"| {f.key} | {f.value} | {f.basis} | `{f.source}` |")
    a("")
    return "\n".join(lines)


def render_model_card(db_path: Path = DUCKDB_PATH, out_path: Path = MODEL_CARD_PATH) -> str:
    """Render the model card from ``db_path`` (the DuckDB file) to ``out_path``.

    Every number in the card must be read from the database or from a
    recorded measurement artifact (``hmda verify --card`` checks this). Must
    include :data:`NO_CREDIT_DECISIONS_SENTENCE` verbatim. Returns the
    rendered text.
    """
    facts = collect_facts(db_path=db_path)
    text = _card_text(facts)

    if NO_CREDIT_DECISIONS_SENTENCE not in text:
        raise ValueError("model card: the no-credit-decisions sentence was not rendered (C-37)")
    banned = banned_terms_in(text)
    if banned:
        raise ValueError(f"model card: banned abbreviation(s) in rendered text: {banned}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify_card(
    card_path: Path = MODEL_CARD_PATH,
    db_path: Path = DUCKDB_PATH,
    numbers_path: Path = RECORDED_RUNS_PATH,
    controls_path: Path = CONTROLS_PATH,
) -> list[str]:
    """Return a list of problems with the shipped card; empty means it verifies.

    Checks, in order: the card exists; it carries the C-37 sentence
    verbatim; it carries no banned abbreviation; and every numeric token in
    it is a token of a currently measured fact. The last check is what
    catches a hand-edited number.
    """
    problems: list[str] = []
    if not card_path.is_file():
        return [f"{card_path} does not exist; run: hmda report --model-card"]

    text = card_path.read_text(encoding="utf-8")

    if NO_CREDIT_DECISIONS_SENTENCE not in text:
        problems.append(
            f"C-37 VIOLATED: {card_path} does not contain "
            f"{NO_CREDIT_DECISIONS_SENTENCE!r} verbatim"
        )

    for term in banned_terms_in(text):
        problems.append(f"banned abbreviation {term!r} appears in {card_path}")

    facts = collect_facts(db_path=db_path, numbers_path=numbers_path, controls_path=controls_path)
    allowed: set[str] = set()
    for fact in facts.values():
        allowed.update(number_tokens(fact.value))
        allowed.update(number_tokens(fact.basis))
        allowed.update(number_tokens(fact.source))

    for token in sorted(set(number_tokens(text))):
        if token not in allowed:
            problems.append(
                f"{card_path}: the number {token!r} is not a measured fact. "
                "Either it was hand-edited, or the source of truth moved. "
                "Regenerate with: hmda report --model-card"
            )
    return problems


# --------------------------------------------------------------------------
# Entry points (what the CLI calls)
# --------------------------------------------------------------------------
def render_main(db_path: Path = DUCKDB_PATH, out_path: Path = MODEL_CARD_PATH) -> int:
    """``hmda report --model-card``: regenerate the card. Returns a process exit code."""
    if not Path(db_path).exists():
        print(f"hmda report --model-card: {db_path} does not exist yet", file=sys.stderr)
        return 1
    try:
        text = render_model_card(db_path=Path(db_path), out_path=Path(out_path))
    except (ValueError, OSError) as exc:
        print(f"hmda report --model-card: {exc}", file=sys.stderr)
        return 1
    facts = collect_facts(db_path=Path(db_path))
    print(f"wrote {out_path}: {len(text.splitlines())} lines, {len(facts)} measured facts")
    return 0


def verify_main(card_path: Path = MODEL_CARD_PATH, db_path: Path = DUCKDB_PATH) -> int:
    """``hmda verify --card``: check every number in the card. Returns a process exit code."""
    if not Path(db_path).exists():
        print(f"hmda verify --card: {db_path} does not exist yet", file=sys.stderr)
        return 1
    try:
        problems = verify_card(card_path=Path(card_path), db_path=Path(db_path))
    except (ValueError, OSError) as exc:
        print(f"hmda verify --card: {exc}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"hmda verify --card: FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"hmda verify --card: PASS ({card_path})")
    return 0
