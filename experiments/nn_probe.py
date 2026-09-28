"""A small neural net for denial risk, and probes for race it was never given.

    .venv/bin/python experiments/nn_probe.py [--n 500000] [--seeds 0 1 2 3 4]

Needs the optional torch extra: uv pip install -e ".[nn]".

Question: the model's inputs come from the feature ALLOWLIST
(hmda.model.features.FEATURE_COLUMNS), so race is never an input. Can race
still be read out of the inputs and out of the network's hidden layers? If a
probe predicts Black vs White applicants well, "we removed the race column"
does not make the model race-blind: the information arrives through proxies
(income, loan size, property value, ...).

A hidden layer is a function of the inputs, so it cannot hold MORE race
information than the inputs do. What the probes measure is how easy that
information is to read out, linearly or with a small nonlinear model.

Per seed (the seed sets both the national sample draw and the weight init):
  - 2023-2024 rows are split 90/10 at random into train and validation. The
    imputer, scaler, network, logistic regression and LightGBM are all fitted
    on the same 90%. The network trains up to 30 epochs with early stopping
    on validation loss (patience 4); the best epoch's weights are restored.
  - 2025 rows are the test set. Denial AUC is scored there.
  - Black and White test rows are split in half once. Every probe is fitted
    on the first half and scored by AUC on the second half. Two probe kinds:
    logistic regression (linear) and a one-hidden-layer MLP (64 units),
    early-stopped on held-out log loss (see mlp_probe for why not accuracy).
  - Representations probed: input features, hidden layer 1, hidden layer 2,
    and layer 2 of a random-init network (control: same architecture,
    untrained). A probe on random-init layer 2 is in effect a probe on random
    ReLU features of the inputs.
  - Shuffled-label control on hidden layer 2, for both probe kinds; should be
    about 0.5.

Extra analysis: which input features carry the race signal. Chosen over the
"race direction vs denial score" option because it answers a question a
reviewer can act on (which allowed features are proxies) and it is harder to
over-read: a correlation between a race direction and the denial score would
be confounded by the same proxies and invites a causal reading this design
cannot support. Standardized coefficients of the linear input probe are
reported, averaged over seeds with a count of how many seeds each feature was
in the top 5. One-hot columns are collinear, so single coefficients are
unstable; each feature's one-feature AUC (direction-free, max(a, 1 - a)) is
reported too as a check that does not depend on the other columns.

Every number is a SAMPLE number. This measures what information is present
in the inputs and representations, not intent, and not legal disparate
treatment. Per-seed detail: out/nn_probe_seed<k>.json (gitignored).
Aggregate plus per-seed detail: results/nn_probe.json (committed).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from hmda.clean.source import NATIONAL, open_source
from hmda.model import features as F
from hmda.model.gbm import fit_gbm, fit_logistic_regression

ROOT = Path(__file__).resolve().parents[1]
GROUP, REFERENCE = "Black or African American", "White"
MAX_EPOCHS, PATIENCE, VAL_SHARE = 30, 4, 0.10

# Code labels from the CFPB HMDA LAR data field reference,
# https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
# (fetched 2026-09-28). Used only to print readable feature names.
_CODE_LABELS = {
    "loan_type": {"1": "Conventional", "2": "FHA insured", "3": "VA guaranteed", "4": "RHS or FSA guaranteed"},
    "loan_purpose": {"1": "Home purchase", "2": "Home improvement", "31": "Refinancing",
                     "32": "Cash-out refinancing", "4": "Other purpose", "5": "Not applicable"},
    "occupancy_type": {"1": "Principal residence", "2": "Second residence", "3": "Investment property"},
    "lien_status": {"1": "First lien", "2": "Subordinate lien"},
    "preapproval": {"1": "Preapproval requested", "2": "Preapproval not requested"},
    "conforming_loan_limit": {"C": "Conforming", "NC": "Nonconforming", "U": "Undetermined", "NA": "Not applicable"},
    "construction_method": {"1": "Site-built", "2": "Manufactured home"},
    "open-end_line_of_credit": {"1": "Open-end line of credit", "2": "Not open-end", "1111": "Exempt"},
    "business_or_commercial_purpose": {"1": "Business purpose", "2": "Not business purpose", "1111": "Exempt"},
    "reverse_mortgage": {"1": "Reverse mortgage", "2": "Not reverse mortgage", "1111": "Exempt"},
}
_NUMERIC_LABELS = {
    "loan_amount": "Loan amount", "income_dollars": "Income", "loan_to_value_ratio": "Loan-to-value ratio",
    "property_value": "Property value", "debt_to_income_rank": "Debt-to-income band",
    "loan_to_income_ratio": "Loan-to-income ratio",
}


def readable(name: str) -> str:
    if name.startswith("missingindicator_"):
        return readable(name[len("missingindicator_"):]) + " (missing)"
    if name in _NUMERIC_LABELS:
        return _NUMERIC_LABELS[name]
    for col in sorted(_CODE_LABELS, key=len, reverse=True):
        if name.startswith(col + "_"):
            code = name[len(col) + 1:]
            return f"{col.replace('_', ' ')}: {_CODE_LABELS[col].get(code, code)}"
    if name.startswith("total_units_"):
        return f"total units: {name[len('total_units_'):]}"
    return name


class Net(torch.nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return self.out(h2).squeeze(-1), h1, h2


@torch.no_grad()
def run_net(net, X, device):
    logits, h1, h2 = net(torch.tensor(X, dtype=torch.float32).to(device))
    return torch.sigmoid(logits).cpu().numpy(), h1.cpu().numpy(), h2.cpu().numpy()


@torch.no_grad()
def val_loss(net, X, y, device):
    logits, _, _ = net(torch.tensor(X, dtype=torch.float32).to(device))
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logits, yt).item())


def train_net(X, y, Xv, yv, seed, device, batch=4096):
    """Adam, early stopping on validation loss, best weights restored."""
    torch.manual_seed(seed)
    net = Net(X.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, history = float("inf"), 0, None, []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            logits, _, _ = net(Xt[idx].to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        vl = val_loss(net, Xv, yv, device)
        history.append(round(vl, 5))
        print(f"  epoch {ep}: validation loss {vl:.4f}")
        if vl < best - 1e-5:
            best, best_ep, best_state = vl, ep, copy.deepcopy(net.state_dict())
        elif ep - best_ep >= PATIENCE:
            break
    net.load_state_dict(best_state)
    return net, {"best_epoch": best_ep, "epochs_run": len(history), "best_val_loss": best, "val_loss_history": history}


def linear_probe(Za, la, Zb, lb):
    sc = StandardScaler().fit(Za)
    p = LogisticRegression(max_iter=3000).fit(sc.transform(Za), la)
    return float(roc_auc_score(lb, p.predict_proba(sc.transform(Zb))[:, 1])), p


def mlp_probe(Za, la, Zb, lb, seed, max_epochs=100, patience=5):
    """One-hidden-layer MLP probe, early-stopped on validation LOG LOSS.

    sklearn's built-in early_stopping monitors validation accuracy, which at
    an 11% positive rate barely moves and stopped the probe near a majority
    classifier in a smoke test. So 10% of the fitting half is held out, one
    epoch runs per partial_fit call, and the epoch with the lowest held-out
    log loss is kept. The scoring half is never seen during fitting.
    """
    sc = StandardScaler().fit(Za)
    Xa = sc.transform(Za)
    rng = np.random.default_rng(seed)
    hold = rng.random(len(Xa)) < 0.1
    p = MLPClassifier(hidden_layer_sizes=(64,), random_state=seed, batch_size=256, learning_rate_init=1e-3)
    best, best_model, since = float("inf"), None, 0
    for _ in range(max_epochs):
        p.partial_fit(Xa[~hold], la[~hold], classes=np.array([0, 1]))
        vl = log_loss(la[hold], p.predict_proba(Xa[hold])[:, 1], labels=[0, 1])
        if vl < best - 1e-5:
            best, best_model, since = vl, copy.deepcopy(p), 0
        else:
            since += 1
            if since >= patience:
                break
    return float(roc_auc_score(lb, best_model.predict_proba(sc.transform(Zb))[:, 1]))


def run_seed(n: int, seed: int, device: str) -> dict:
    t0 = time.time()
    src = open_source(NATIONAL)
    raw = src.sample(n, seed=seed)
    print(f"{src.label}\nSAMPLE: {len(raw):,} rows (seed {seed})")
    analysis = F.analysis_set(raw)
    train_all, test = F.time_split(analysis, 2025)
    X_all, y_all = F.build_feature_matrix(train_all)
    X_te, y_te = F.build_feature_matrix(test)
    X_te = X_te.reindex(columns=X_all.columns, fill_value=0.0)
    assert not set(X_all.columns) & {"derived_race", "derived_ethnicity", "derived_sex"}

    rng = np.random.default_rng(seed)
    is_val = rng.random(len(X_all)) < VAL_SHARE
    X_tr, y_tr = X_all.loc[~is_val], y_all.loc[~is_val]
    X_va, y_va = X_all.loc[is_val], y_all.loc[is_val]
    print(f"train {len(X_tr):,} / validation {len(X_va):,} (2023-2024), test {len(X_te):,} (2025), "
          f"{X_all.shape[1]} features")

    imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))
    names = [str(c) for c in imp.get_feature_names_out()]
    prep = lambda X: sc.transform(imp.transform(X)).astype(np.float32)  # noqa: E731
    A_tr, A_va, A_te = prep(X_tr), prep(X_va), prep(X_te)

    print(f"training the network on {device}")
    net, training = train_net(A_tr, y_tr.to_numpy(), A_va, y_va.to_numpy(), seed, device)
    p_net, h1, h2 = run_net(net, A_te, device)
    auc = {"neural_net": float(roc_auc_score(y_te, p_net))}
    print("fitting the repo's logistic regression and LightGBM on the same 90% training rows")
    auc["logistic_regression"] = float(roc_auc_score(y_te, fit_logistic_regression(X_tr, y_tr).predict_proba(X_te)[:, 1]))
    auc["lightgbm"] = float(roc_auc_score(y_te, fit_gbm(X_tr, y_tr).predict_proba(X_te)[:, 1]))

    race = test["derived_race"].to_numpy()
    keep = np.isin(race, [GROUP, REFERENCE])
    lab = (race[keep] == GROUP).astype(int)
    torch.manual_seed(seed + 1000)
    _, _, h2_rand = run_net(Net(A_tr.shape[1]).to(device), A_te, device)
    reps = {"input_features": A_te[keep], "hidden_layer_1": h1[keep], "hidden_layer_2": h2[keep],
            "control_random_init_layer_2": h2_rand[keep]}

    perm = rng.permutation(len(lab))
    a, b = perm[: len(lab) // 2], perm[len(lab) // 2 :]
    shuffled = rng.permutation(lab)
    linear, mlp = {}, {}
    input_probe = None
    for k, Z in reps.items():
        print(f"  probing {k}")
        linear[k], model = linear_probe(Z[a], lab[a], Z[b], lab[b])
        if k == "input_features":
            input_probe = model
        mlp[k] = mlp_probe(Z[a], lab[a], Z[b], lab[b], seed)
    Z = reps["hidden_layer_2"]
    linear["control_shuffled_labels_layer_2"], _ = linear_probe(Z[a], shuffled[a], Z[b], shuffled[b])
    mlp["control_shuffled_labels_layer_2"] = mlp_probe(Z[a], shuffled[a], Z[b], shuffled[b], seed)
    denial_score_only = float(roc_auc_score(lab, p_net[keep]))

    # Extra: which inputs carry the race signal. Coefficients are on
    # standardized inputs (the probe re-standardizes), positive = more Black.
    coef = input_probe.coef_[0]
    Zb = reps["input_features"][b]
    single = {}
    for j, name in enumerate(names):
        if np.ptp(Zb[:, j]) == 0:
            continue
        s = roc_auc_score(lab[b], Zb[:, j])
        single[name] = float(max(s, 1 - s))
    features = {name: {"coef": float(coef[j]), "single_feature_auc": single.get(name)} for j, name in enumerate(names)}

    return {
        "seed": seed, "sample_rows": len(raw), "rows_train": len(X_tr), "rows_validation": len(X_va),
        "rows_test": len(X_te), "n_features_after_imputation": len(names),
        "probe_rows_black_white": int(keep.sum()), "black_share": float(lab.mean()),
        "training": training, "denial_auc": auc,
        "race_probe_auc_linear": linear, "race_probe_auc_mlp": mlp,
        "denial_score_only_auc": denial_score_only,
        "input_features": features,
        "seconds": round(time.time() - t0, 1),
        "source": src.label.replace(str(ROOT) + "/", ""),
    }


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def aggregate(runs: list[dict], n: int) -> dict:
    agg = {"denial_auc": {}, "race_probe_auc_linear": {}, "race_probe_auc_mlp": {}}
    for block in agg:
        for k in runs[0][block]:
            agg[block][k] = mean_std([r[block][k] for r in runs])
    agg["denial_score_only_auc"] = mean_std([r["denial_score_only_auc"] for r in runs])
    agg["best_epoch"] = mean_std([r["training"]["best_epoch"] for r in runs])
    agg["seeds_where_early_stopping_fired"] = sum(r["training"]["epochs_run"] < MAX_EPOCHS for r in runs)
    # Paired by seed: same sample, same probe split, so the seed-to-seed noise
    # that is shared by both sides cancels.
    pairs = [("input_features", "hidden_layer_1"), ("hidden_layer_1", "hidden_layer_2"),
             ("hidden_layer_2", "control_random_init_layer_2")]
    agg["paired_differences"] = {}
    for kind in ("linear", "mlp"):
        blk = f"race_probe_auc_{kind}"
        for x, y in pairs:
            agg["paired_differences"][f"{kind}: {x} minus {y}"] = mean_std([r[blk][x] - r[blk][y] for r in runs])
    for k in ("input_features", "hidden_layer_1", "hidden_layer_2", "control_random_init_layer_2"):
        agg["paired_differences"][f"mlp minus linear: {k}"] = mean_std(
            [r["race_probe_auc_mlp"][k] - r["race_probe_auc_linear"][k] for r in runs])

    names = runs[0]["input_features"].keys()
    top5 = {}
    for r in runs:
        ranked = sorted(r["input_features"].items(), key=lambda kv: -abs(kv[1]["coef"]))[:5]
        for name, _ in ranked:
            top5[name] = top5.get(name, 0) + 1
    rows = []
    for name in names:
        coefs = [r["input_features"][name]["coef"] for r in runs]
        singles = [r["input_features"][name]["single_feature_auc"] for r in runs
                   if r["input_features"][name]["single_feature_auc"] is not None]
        rows.append({"feature": name, "readable": readable(name),
                     "coef": mean_std(coefs), "seeds_in_top5_by_abs_coef": top5.get(name, 0),
                     "single_feature_auc": mean_std(singles) if singles else None})
    by_coef = sorted(rows, key=lambda r: -abs(r["coef"]["mean"]))
    by_single = sorted([r for r in rows if r["single_feature_auc"]], key=lambda r: -r["single_feature_auc"]["mean"])

    return {
        "status": "SAMPLE_BASED",
        "command": "python experiments/nn_probe.py --n %d --seeds %s" % (n, " ".join(str(r["seed"]) for r in runs)),
        "source": runs[0]["source"],
        "note": ("Every number is a SAMPLE number from a stratified national sample, one draw per seed. "
                 "Probe AUCs measure how readable Black vs White applicant status is from inputs or hidden "
                 "layers; they do not measure intent or legal disparate treatment. Std uses ddof=1."),
        "design": {"sample_rows_requested": n, "seeds": [r["seed"] for r in runs], "split": "train 2023-2024 (90%), "
                   "validation 2023-2024 (10%, random), test 2025", "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "network": "inputs -> 64 -> 32 -> 1, ReLU, Adam lr 1e-3", "probe_split": "Black/White test rows, "
                   "fit on one half, AUC on the other", "linear_probe": "LogisticRegression on standardized features",
                   "mlp_probe": ("MLPClassifier(hidden_layer_sizes=(64,)) on standardized features, one epoch per partial_fit, "
                                 "early-stopped on log loss of a 10% hold-out of the fitting half (patience 5)")},
        "aggregate": agg,
        "input_feature_race_signal": {
            "method": ("Standardized coefficients of the linear probe on input features (positive = higher odds the "
                       "applicant is Black), and each feature's one-feature AUC, max(auc, 1 - auc). One-hot columns "
                       "are collinear, so single coefficients can split across dummies; read the one-feature AUC as "
                       "the more stable number."),
            "top10_by_abs_coef": by_coef[:10],
            "top10_by_single_feature_auc": by_single[:10],
        },
        "per_seed": [{k: v for k, v in r.items() if k not in ("input_features", "source")} for r in runs],
        "seconds_total": round(sum(r["seconds"] for r in runs), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--reuse", action="store_true", help="reuse out/nn_probe_seed<k>.json if present")
    args = ap.parse_args(argv)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (ROOT / "out").mkdir(exist_ok=True)
    runs = []
    for s in args.seeds:
        path = ROOT / "out" / f"nn_probe_seed{s}.json"
        if args.reuse and path.exists():
            runs.append(json.loads(path.read_text()))
            continue
        r = run_seed(args.n, s, device)
        path.write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps({k: r[k] for k in ("denial_auc", "race_probe_auc_linear", "race_probe_auc_mlp")}, indent=2))
        runs.append(r)
    out = aggregate(runs, args.n)
    dest = ROOT / "results" / "nn_probe.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["aggregate"], indent=2))
    print(f"aggregate -> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
