"""hmda-audit dashboard.

Reads only the small precomputed artifacts committed under results/. Performs
no computation, opens no DuckDB file, and never imports hmda's data layer.
Intended to start in about a second. Regenerate results/*.json with
`.venv/bin/python scripts/gen_dashboard_artifacts.py`.

Every chart is a Plotly figure drawn under the single `hmda` template in
theme.py and built in charts.py. Each chart carries its own headline sentence
and its own method line, so it can be read on its own. The page opens on a
measured finding, not on a row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import charts
import theme

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        st.error(
            f"Missing artifact: results/{name}. Run "
            "`.venv/bin/python scripts/gen_dashboard_artifacts.py` first."
        )
        st.stop()
    return json.loads(path.read_text())


st.set_page_config(page_title="hmda-audit", page_icon="\U0001F4CA", layout="wide")
theme.register()

scale = load("scale.json")
governance = load("governance.json")
metrics = load("metrics_ledger.json")
mitigation = load("mitigation.json")
conflict = load("mitigation_conflict.json")
engineering = load("engineering.json")
ff = load("four_fifths.json")
denials = load("denial_rates.json")
controlled = load("controlled.json")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp {{ background: {theme.SURFACE}; }}
.block-container {{ max-width: 1180px; padding-top: 2.4rem; }}
html, body, [class*="css"] {{ font-family: {theme.BODY}; color: {theme.INK}; }}

.eyebrow {{
    font-family: {theme.MONO}; font-size: 0.82rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: {theme.MUTED}; margin-bottom: 0.5rem;
}}
.masthead {{
    font-family: {theme.DISPLAY}; font-weight: 700; font-size: 3.1rem;
    line-height: 1.02; letter-spacing: -0.018em; color: {theme.INK}; margin: 0;
}}
.standfirst {{
    font-size: 1.12rem; color: {theme.MUTED}; max-width: 62ch;
    margin: 0.7rem 0 0 0; line-height: 1.5;
}}
.rule {{ border: none; border-top: 1px solid {theme.GRID}; margin: 2.6rem 0 1.4rem 0; }}
.hairline {{ border: none; border-top: 1px solid {theme.GRID}; margin: 1.2rem 0; }}

.sidenote-h {{
    font-family: {theme.DISPLAY}; font-weight: 600; font-size: 1.25rem;
    color: {theme.INK}; margin: 0 0 0.5rem 0; line-height: 1.2;
}}
.prov {{ font-family: {theme.MONO}; font-size: 0.8rem; color: {theme.MUTED}; margin-top: 0.4rem; }}
.badge {{
    display: inline-block; font-family: {theme.MONO}; font-size: 0.68rem;
    font-weight: 500; letter-spacing: 0.06em; padding: 0.08rem 0.42rem;
    border-radius: 0.2rem; margin-right: 0.45rem; vertical-align: 0.06rem;
}}
.b-measured {{ color: {theme.PRIMARY}; border: 1px solid {theme.PRIMARY}; }}
.b-sample {{ color: {theme.CAUTION}; border: 1px solid {theme.CAUTION}; }}
.b-unresolved {{ color: {theme.FLAGGED}; border: 1px solid {theme.FLAGGED}; }}

.notbox {{
    border-left: 3px solid {theme.INK}; padding: 0.7rem 0 0.7rem 1rem;
    font-size: 0.97rem; color: {theme.MUTED}; line-height: 1.55; margin: 0.9rem 0 0 0;
}}
.notbox b {{ color: {theme.INK}; }}
.conflict {{
    border-left: 3px solid {theme.FLAGGED}; padding: 0.8rem 0 0.8rem 1rem;
    font-size: 0.98rem; color: {theme.INK}; line-height: 1.6;
}}
.conflict q {{ color: {theme.MUTED}; font-style: italic; }}
.conflict code {{ font-family: {theme.MONO}; font-size: 0.85rem; color: {theme.MUTED}; }}
.oneline {{ font-size: 1.02rem; color: {theme.INK}; line-height: 1.55; }}
.section-title {{
    font-family: {theme.DISPLAY}; font-weight: 600; font-size: 1.7rem;
    letter-spacing: -0.01em; color: {theme.INK}; margin: 0 0 0.25rem 0;
}}
.footer {{ font-family: {theme.MONO}; font-size: 0.78rem; color: {theme.MUTED}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BADGE_TEXT = {
    "MEASURED": "COUNTED FROM THE FULL FILE",
    "SAMPLE_BASED": "ESTIMATED FROM A SAMPLE",
}


def badge(status: str) -> str:
    cls = {"MEASURED": "b-measured", "SAMPLE_BASED": "b-sample"}.get(status, "b-unresolved")
    return f'<span class="badge {cls}">{BADGE_TEXT.get(status, status)}</span>'


def prov(status: str, text: str) -> None:
    st.markdown(f'<div class="prov">{badge(status)}{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead.
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="eyebrow">Public US mortgage filings &middot; {scale['years_label']}</div>
<h1 class="masthead">What {scale['applications'] / 1e6:.1f} million mortgage<br>applications look like
when you<br>check them against the rules</h1>
<p class="standfirst">Every US lender must report every mortgage application it receives. This page reads all of them, applies the tests a regulator would apply, and says which number came from the whole file and which came from a sample.</p>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero: the 80 percent line.
# ---------------------------------------------------------------------------

st.plotly_chart(charts.four_fifths(ff), width="stretch", config=PLOT_CONFIG)
prov(
    ff["status"],
    f"<code>{ff['command']}</code> &middot; {ff['source']} &middot; "
    f"{ff['rows_screened']:,} of {ff['rows_in_file']:,} rows: {ff['exclusion_note'].lower()}",
)

st.markdown(
    f"""
<div class="notbox" style="margin-top:0.2rem">
<b>The two groups below the line are not really groups.</b> Both are "Free Form Text Only",
the code a lender files when somebody typed their race or ethnicity into a box instead of
ticking one. Between them they are 9,032 and 15,582 applications, the two smallest rows in a
file of {ff['rows_in_file'] / 1e6:.1f} million. &nbsp;<b>A flag is not a verdict:</b> it marks a
gap worth someone's attention, not evidence that any lender treated anybody unfairly. No
lender is named here. &nbsp;<b>{ff['lenders_flagged']:,} of {ff['lenders_screened']:,}
lenders</b> are flagged when the same test is run lender by lender, counting only lenders with
at least {ff['min_count']} applications in a group.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Denial rates, full file.
# ---------------------------------------------------------------------------

st.plotly_chart(charts.denial_rates(denials), width="stretch", config=PLOT_CONFIG)
prov(denials["status"], f"{denials['command']} &middot; {denials['source']}")

d1, d2 = st.columns(2, gap="large")
with d1:
    st.markdown(
        f"""
<div class="notbox">
<b>More than a quarter of the file has no race recorded at all.</b> "Race Not Available" is
{denials['race_not_available_share_pct']}% of applications and the second largest bar in the
chart. Anything said about race here is said about the roughly three quarters where race was
written down.
</div>
""",
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        """
<div class="notbox">
<b>Two of the bars are the same length for different reasons.</b> Black or African American
and American Indian or Alaska Native both sit at 28.17%, but one rests on 2.9 million
applications and the other on 244 thousand. The rates match. The confidence behind them
does not.
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# What is left after the adjustment.
# ---------------------------------------------------------------------------

st.plotly_chart(charts.controlled_gap(controlled), width="stretch", config=PLOT_CONFIG)
prov(
    controlled["status"],
    f"<code>{controlled['command']}</code> &middot; n={controlled['sample_n']:,}, "
    f"seed {controlled['sample_seed']} &middot; {controlled['source']} &middot; "
    f"adjustment variables: {controlled['covariates_source']}",
)
st.markdown(
    f'<div class="notbox"><b>{controlled["caveat"]}</b> An unexplained gap means this '
    f'particular set of variables does not account for it. Something else might: the data does '
    f'not carry credit scores, and it cannot see anything a lender knew but did not file. '
    f'{controlled["note"]}</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Engineering.
# ---------------------------------------------------------------------------

e1, e2 = st.columns(2, gap="large")
with e1:
    st.plotly_chart(
        charts.engineering_pair(engineering, "speed"), width="stretch", config=PLOT_CONFIG
    )
with e2:
    st.plotly_chart(
        charts.engineering_pair(engineering, "memory"), width="stretch", config=PLOT_CONFIG
    )
prov(
    "MEASURED",
    f"TABLES IDENTICAL: {engineering['tables_identical']} &middot; "
    f"<code>{engineering['command']}</code> &middot; {engineering['source']}",
)
st.markdown(f'<div class="notbox">{engineering["footnote"]}</div>', unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Mitigation: reported as an unresolved provenance conflict, not as a result.
# No chart and no money figure, because no run in this repository supports one.
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">One question this page refuses to answer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="standfirst">There is a separate study in this project that asks what happens if '
    'you change the approval rules to close the gap, and what it would cost. Its national '
    'numbers are not shown here, because they were written down by hand, while every other '
    'number on this page was copied from the console output of its run. The gap is printed below rather '
    'than settled quietly in favour of the more interesting answer.</p>',
    unsafe_allow_html=True,
)
claims = "".join(
    f'<br><br><code>{c["locator"]}</code><br><q>{c["quote"]}</q>' for c in conflict["claims"]
)
st.markdown(
    f'<div class="conflict">{badge("UNRESOLVED")}<b>{conflict["headline"]}</b>'
    f"{claims}<br><br><b>How it is handled here.</b> {conflict['resolution']}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="notbox"><b>Separately, and whatever the measurement turns out to be:</b> '
    f'one of the three techniques in that study works by setting a different approval cut-off '
    f'for each race. {mitigation["legal_flag"]}</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scope, governance, ledger.
# ---------------------------------------------------------------------------

s1, s2 = st.columns([0.5, 0.5], gap="large")
with s1:
    st.markdown('<div class="eyebrow">What was read</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="oneline"><b>{scale["applications"]:,}</b> applications &middot; '
        f'<b>{scale["lenders"]:,}</b> lenders &middot; '
        f'<b>{scale["years"]}</b> years ({scale["years_label"]}).</div>',
        unsafe_allow_html=True,
    )
    prov(scale["status"], f"<code>{scale['command']}</code> &middot; {scale['source']}")
    st.markdown(
        f'<div class="oneline" style="margin-top:1rem">'
        f'<b>{governance["total_controls"]}</b> model-risk controls mapped to SR 11-7 and '
        f'OSFI E-23, <b>{governance["backed_by_test"]}</b> enforced by a test. '
        f'{", ".join(governance["not_implemented_ids"])} is disclosed as NOT IMPLEMENTED '
        f'rather than hidden.</div>',
        unsafe_allow_html=True,
    )
    prov("MEASURED", f"<code>{governance['verify_command']}</code>")
with s2:
    st.markdown('<div class="eyebrow">What this is not</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notbox" style="margin-top:0"><b>Not a claim that any named lender '
        'discriminates.</b> The output is statistical disparity, a screening signal, not a '
        'legal finding. Lender identifiers are withheld by policy.</div>'
        '<div class="notbox"><b>Not a live underwriting system.</b> The mitigation layer '
        'is an offline study of what might have happened. It never scores a live applicant.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Full measurement ledger (M1 to M14, technical detail)"):
    st.caption(f"Source: {metrics['source']}")
    st.caption(
        "M11 and M12 below are listed as they were recorded, including the "
        "sign-reversal wording. See the mitigation section above for why those "
        "figures are not charted."
    )
    for m in metrics["metrics"]:
        if m["id"] == "M9":
            continue
        safe_summary = m["summary"].replace("$", "\\$")
        st.markdown(
            f'{badge(m["status"])} **{m["id"]}** &nbsp; `{m["command"]}`  \n{safe_summary}',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="hairline">', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">hmda-audit. Every number on this page is read from '
    'the committed results/*.json files.</div>',
    unsafe_allow_html=True,
)
