"""
make_figures.py
===============
Builds the figures from `results/` and the processed panel.

    python src/make_figures.py              # the four default figures
    python src/make_figures.py --all        # also convergence and forecast example
    python src/make_figures.py --forecast   # also the forecast example only

Default figures
---------------
    fig1_heterogeneity.png   scale, shape and fare-mix skew under the route partition
    fig2_ablation.png        collaboration benefit vs mean EMD, all partitions and windows
    fig3_arms.png            arm comparison, including the gradient boosting baseline
    fig4_client_gain.png     per-client benefit of collaboration vs client size

Optional (`--all`, or `--forecast` for the second):
    convergence.png          validation MAE vs communication round
    forecast_example.png     predicted vs actual over the test week

Where the numbers come from
---------------------------
Every number in a title, axis label or annotation is read from
`results/paper_numbers.json` (written by `src/paper_numbers.py`), so figures
and text use the same values. Plotted series (scatter points, profiles) are
read from the panel and from `per_client.csv`.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataset as ds
import month_config as mc
import paper_numbers as pn

PALETTE = {
    "isolated": "#c44e52",
    "gbm_isolated": "#dd8452",
    "centralised": "#4c72b0",
    "fedavg": "#55a868",
    "fedprox": "#8172b2",
    "dp_fedavg": "#937860",
    "fedavg_finetuned": "#2f7a4f",
}
BASELINE_COLOR = "#9a9a9a"

PRETTY = {
    "isolated": "Isolated MLP\n(no collaboration)",
    "gbm_isolated": "Isolated GBM\n(no collaboration)",
    "centralised": "Centralised\n(pooled)",
    "fedavg": "FedAvg\n(Flower)",
    "fedprox": "FedProx\n(Flower)",
    "dp_fedavg": "DP-FedAvg\n(Flower)",
    "fedavg_finetuned": "FedAvg\n+ fine-tuning",
    "seasonal_naive_1w": "Seasonal naive\n(same hour, −1 week)",
    "persistence_1d": "Persistence\n(same hour, −1 day)",
    "hour_mean_7d": "Same-hour mean\n(7-day)",
}

SCHEME_PRETTY = {
    "route": "route\n(structural)",
    "random": "random",
    "dirichlet_0.1": "Dirichlet\nα=0.1",
    "dirichlet_0.5": "Dirichlet\nα=0.5",
    "dirichlet_5.0": "Dirichlet\nα=5.0",
}

ABLATION_ARMS = ["isolated", "fedavg", "fedavg_finetuned"]
FARE_COLS = ["n_school", "n_pass", "n_single", "n_privilege"]

# Drawn on every run.
DEFAULT_FIGURES = ["fig1_heterogeneity.png", "fig2_ablation.png",
                   "fig3_arms.png", "fig4_client_gain.png"]

# Drawn only on --all / --forecast.
OPTIONAL_FIGURES = ["convergence.png", "forecast_example.png"]


def _tidy(ax) -> None:
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _train_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """
    The training partitions, the same basis `paper_numbers.py` uses for the
    annotated quantities.
    """
    return panel[panel["day"].isin(ds.TRAIN_DAYS)]


# ---------------------------------------------------------------------------
# fig 1: heterogeneity
# ---------------------------------------------------------------------------
def fig_heterogeneity(numbers: dict, panel_path: str, out: str) -> None:
    """Scale, shape and fare-mix skew under the route partition."""
    c = numbers["clients"]
    panel = ds.load_panel(panel_path)
    train = _train_rows(panel)

    means = pd.Series(c["mean_boardings_by_client"], dtype=float).sort_values(
        ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) scale skew
    ax = axes[0]
    ax.hist(means.values,
            bins=np.logspace(np.log10(max(means.min(), 0.1)),
                             np.log10(means.max()), 22),
            color="#4c72b0", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Mean boardings per hour (log)")
    ax.set_ylabel("Number of clients")
    ax.set_title(f"(a) Scale skew — {c['scale_ratio']:.0f}× spread\n"
                 f"{c['min_client_mean_boardings']:.1f}/h (route "
                 f"{c['min_client']}) to {c['max_client_mean_boardings']:.0f}/h "
                 f"(route {c['max_client']})", fontsize=10)

    # (b) shape skew: peak-normalised hourly profile for four clients spanning
    # the size range
    ax = axes[1]
    picks = [means.index[0], means.index[len(means) // 4],
             means.index[len(means) // 2], means.index[-8]]
    for r in picks:
        prof = train[train["route"] == r].groupby("hour")[ds.TARGET].mean()
        ax.plot(prof.index, prof / prof.max(), lw=2,
                label=f"route {r} ({means[r]:.0f}/h)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Boardings (normalised to own peak)")
    ax.set_title("(b) Shape skew — different demand rhythms\n"
                 f"mean pairwise profile distance {c['shape_divergence']:.3f}",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # (c) fare-mix skew
    ax = axes[2]
    mix = train.groupby("route")[FARE_COLS].sum()
    mix = mix.div(mix.sum(axis=1), axis=0).reindex(means.index)
    bottom = np.zeros(len(mix))
    for col, colour, lab in zip(
        FARE_COLS,
        ["#dd8452", "#4c72b0", "#55a868", "#c44e52"],
        ["school pass", "travel pass", "single ticket", "concession"],
    ):
        ax.bar(range(len(mix)), mix[col], bottom=bottom, color=colour,
               width=1.0, label=lab)
        bottom += mix[col].values
    ax.set_xlim(-0.5, len(mix) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Clients, ordered by size")
    ax.set_ylabel("Share of boardings")
    ax.set_title("(c) Fare-mix skew — different passenger populations\n"
                 f"mean total-variation distance from global mix "
                 f"{c['faremix_tv']:.3f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    for a in axes:
        _tidy(a)

    fig.suptitle(
        f"Route-level partitioning yields genuine, non-simulated non-IID "
        f"structure across {c['n_clients']} clients\n"
        f"all three measures computed on {c['definition']}",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 2: partition ablation
# ---------------------------------------------------------------------------
def fig_ablation(cross: dict, out: str) -> None:
    """
    Collaboration benefit against partition heterogeneity.

    Mean EMD on the x axis, FedAvg benefit over isolated training on the y
    axis, one series per observation window, every partition plotted. The
    structural (route) partition is drawn with a distinct marker, since it is
    not produced by the synthetic procedure and is compared against it.
    """
    windows = cross["windows"]
    colours = {"jan": "#4c72b0", "jul": "#dd8452", "oct": "#55a868"}
    names = {"jan": "January", "jul": "July", "oct": "October"}

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axhline(0, color="#333", lw=1.2, zorder=1)

    for tag in windows:
        rows = sorted((r for r in cross["rows"] if r["window"] == tag
                       and r["fedavg_vs_isolated_pct"] is not None),
                      key=lambda r: r["emd"])
        if not rows:
            continue
        c = colours.get(tag, "#8172b2")
        syn = [r for r in rows if r["scheme"] != "route"]
        rte = [r for r in rows if r["scheme"] == "route"]

        o = cross["ordering"].get(tag, {}).get("all", {})
        lbl = names.get(tag, tag)
        if "rho" in o:
            lbl += f"  (ρ={o['rho']:+.2f}, p={o['p_one_sided']:.1e})"

        ax.plot([r["emd"] for r in rows], [r["fedavg_vs_isolated_pct"] for r in rows],
                color=c, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter([r["emd"] for r in syn], [r["fedavg_vs_isolated_pct"] for r in syn],
                   s=58, color=c, edgecolor="white", linewidth=1.1, zorder=3,
                   label=lbl)
        if rte:
            r = rte[0]
            ax.scatter([r["emd"]], [r["fedavg_vs_isolated_pct"]], s=340, color=c,
                       marker="*", edgecolor="#222", linewidth=1.3, zorder=4)
            ax.annotate(f"route\n{names.get(tag, tag)}",
                        (r["emd"], r["fedavg_vs_isolated_pct"]),
                        textcoords="offset points", xytext=(11, -4),
                        fontsize=8.5, color=c, fontweight="bold")

    # Mark partitions that could not train every client.
    drop = [r for r in cross["rows"]
            if not r["full_federation"] and r["fedavg_vs_isolated_pct"] is not None]
    if drop:
        ax.scatter([r["emd"] for r in drop], [r["fedavg_vs_isolated_pct"] for r in drop],
                   s=170, facecolor="none", edgecolor="#c44e52", linewidth=1.3,
                   linestyle=":", zorder=5,
                   label="partition dropped clients (not a full federation)")

    ax.set_xlabel("Partition heterogeneity — mean Earth Mover's Distance\n"
                  "(training partitions only)", fontsize=10.5)
    ax.set_ylabel("Benefit of a shared model (%)\n"
                  "FedAvg vs. isolated training; positive = collaboration helped",
                  fontsize=10.5)
    ax.set_title(
        "The more heterogeneous the partition, the less collaboration helps\n"
        f"{len({r['scheme'] for r in cross['rows']})} partitions × "
        f"{len(windows)} observation windows; ★ marks the structural "
        "(route) partition",
        fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    # Room on the right for the route annotations at the high-EMD end.
    ax.margins(x=0.10)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 3: arm comparison
# ---------------------------------------------------------------------------
def fig_arms(numbers: dict, out: str) -> None:
    """Every arm, plus the non-learned baselines, on one axis."""
    rows = [{"arm": a, "mae": v["mae"]["mean"], "sd": v["mae"]["sd"],
             "n_seeds": v["mae"]["n_seeds"], "learned": True}
            for a, v in numbers["arms"].items()]
    rows += [{"arm": k, "mae": v["mae"], "sd": 0.0, "n_seeds": 1,
              "learned": False}
             for k, v in numbers["baselines"].items()]
    df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    colors = [PALETTE.get(a, BASELINE_COLOR) for a in df["arm"]]

    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.bar(range(len(df)), df["mae"], yerr=df["sd"], capsize=4, color=colors,
           edgecolor="white", linewidth=1.2)
    for i, r in df.iterrows():
        ax.text(i, r["mae"] + r["sd"] + 0.35, f"{r['mae']:.2f}", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(i, 0.45, f"n={int(r['n_seeds'])}", ha="center", fontsize=7.5,
                color="white")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([PRETTY.get(a, a) for a in df["arm"]], fontsize=8)
    ax.set_ylabel("Test MAE (boardings / hour)\nmacro-average over "
                  f"{numbers['clients']['n_clients']} routes")
    ax.set_title(
        "Next-hour demand forecasting — Astana bus depot, January 2025\n"
        "lower is better; error bars are ±1 SD over seeds, n per bar",
        fontsize=11.5, pad=12,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE["fedavg"]),
               plt.Rectangle((0, 0), 1, 1, color=PALETTE["gbm_isolated"]),
               plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR)]
    ax.legend(handles, ["Learned, neural", "Learned, gradient boosting",
                        "Non-learned baseline"], frameon=False, fontsize=9)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig 4: per-client gain
# ---------------------------------------------------------------------------
def fig_client_gain(numbers: dict, results_dir: str, out: str) -> None:
    """
    Per-client benefit of collaboration against client size.

    The scatter is the per-client series; the counts and per-band medians
    come from paper_numbers.json.
    """
    pc = numbers["perclient"]
    if not pc.get("available"):
        raise SystemExit(f"fig4 needs per-client results: {pc.get('note')}")

    raw = pd.read_csv(os.path.join(results_dir, "per_client.csv"),
                      dtype={"route": str})
    iso = raw[raw["arm"] == "isolated"].set_index("route")

    arms = [a for a in ("fedavg", "fedavg_finetuned") if a in pc["arms"]]
    fig, axes = plt.subplots(1, len(arms), figsize=(6.9 * len(arms), 5.4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, arm in zip(axes, arms):
        a = pc["arms"][arm]
        other = raw[raw["arm"] == arm].set_index("route")
        j = pd.DataFrame({"size": iso["mean_boardings"],
                          "mae_iso": iso["mae"],
                          "mae_arm": other["mae"]}).dropna()
        j["gain"] = 100 * (j["mae_iso"] - j["mae_arm"]) / j["mae_iso"]

        ax.scatter(j["size"], j["gain"],
                   c=[PALETTE["fedavg"] if g > 0 else PALETTE["isolated"]
                      for g in j["gain"]],
                   s=42, alpha=0.8, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xscale("log")

        # Median by size band, read from paper_numbers.json.
        centres, medians = [], []
        for b in a["bands"]:
            if b["median_mae_change_pct"] is None or b["n_clients"] < 3:
                continue
            hi = b["hi"] if b["hi"] is not None else float(j["size"].max())
            lo = max(b["lo"], float(j["size"].min()))
            centres.append(np.sqrt(max(lo, 1e-6) * hi))
            medians.append(b["median_mae_change_pct"])
        if centres:
            ax.plot(centres, medians, color="#333", lw=2, marker="o", ms=5,
                    label="median by client size band")
            ax.legend(frameon=False, fontsize=9, loc="lower left")

        ax.set_title(f"{PRETTY[arm].replace(chr(10), ' ')}\n"
                     f"{a['n_improved']}/{a['n_clients']} clients improved vs. "
                     f"training alone (median {a['median_gain_pct']:+.1f}%)",
                     fontsize=10.5)
        ax.set_xlabel("Client size — mean boardings per hour, log scale\n"
                      "(training partitions only)")
        _tidy(ax)

    axes[0].set_ylabel("MAE change vs. isolated training (%)\n"
                       "positive = collaboration helped")
    fig.suptitle(
        "Sharing weights alone costs the busiest clients accuracy; "
        "local adaptation recovers it",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional figures, drawn only with --all / --forecast.
# ---------------------------------------------------------------------------
def fig_convergence(numbers: dict, results_dir: str, out: str) -> None:
    conv = pd.read_csv(os.path.join(results_dir, "convergence.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, grp in conv.groupby("arm"):
        agg = grp.groupby("round")["mae"].agg(["mean", "std"]).reset_index()
        agg = agg[agg["round"] > 0]
        ax.plot(agg["round"], agg["mean"], label=arm, color=PALETTE.get(arm), lw=2)
        ax.fill_between(agg["round"],
                        agg["mean"] - agg["std"].fillna(0),
                        agg["mean"] + agg["std"].fillna(0),
                        color=PALETTE.get(arm), alpha=0.15)

    for arm, style in (("isolated", "--"), ("centralised", ":")):
        if arm in numbers["arms"]:
            ax.axhline(numbers["arms"][arm]["mae"]["mean"], color=PALETTE[arm],
                       ls=style, lw=1.6, label=f"{arm} (final test)")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation MAE (boardings / hour)")
    ax.set_title("Convergence of the federated arms", fontsize=11)
    ax.set_ylim(bottom=max(0, conv[conv["round"] > 1]["mae"].min() - 1))
    ax.legend(frameon=False, fontsize=9)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_forecast(numbers: dict, panel_path: str, out: str) -> None:
    """Predicted vs actual over the test window. Retrains one FedAvg model."""
    import fl_core as fl

    panel = ds.load_panel(panel_path)
    clients = ds.build_clients(panel)
    cfg = fl.FLConfig(rounds=30, seed=0)
    model, _ = fl.federated(clients, cfg)

    order = sorted(clients, key=lambda c: c.mean_boardings, reverse=True)
    picks = [order[0], order[len(order) // 2]]

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 6.5))
    for ax, c in zip(axes, picks):
        grp = panel[panel["route"] == c.name]
        te = grp[grp["day"].isin(ds.TEST_DAYS)].sort_values("ts")
        y_true = te[ds.TARGET].values.astype(float)
        y_pred = np.clip(c.denormalise(fl.predict(model, c.x_test).ravel()), 0, None)

        ax.plot(te["ts"].values, y_true, color="#333", lw=1.6, label="actual")
        ax.plot(te["ts"].values, y_pred, color=PALETTE["fedavg"], lw=1.6,
                ls="--", label="FedAvg forecast")
        mae = float(np.mean(np.abs(y_pred - y_true)))
        ax.set_title(f"Route {c.name} — mean {c.mean_boardings:.0f} "
                     f"boardings/h, test MAE {mae:.1f}", fontsize=10)
        ax.set_ylabel("Boardings / hour")
        ax.legend(frameon=False, fontsize=9)
        _tidy(ax)

    d = numbers["panel"]["split_days"]["test"]
    axes[-1].set_xlabel(f"Test window ({min(d)}–{max(d)} January 2025)")
    fig.suptitle("Global federated model, applied per route — "
                 "no raw data ever pooled", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel, results and figures dirs")
    ap.add_argument("--results", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--forecast", action="store_true",
                    help="also draw the forecast example (retrains one FedAvg model)")
    ap.add_argument("--all", action="store_true",
                    help="also draw the convergence and forecast figures")
    args = ap.parse_args()
    paths = mc.paths_for(args.tag)
    args.results = str(ds.resolve_path(args.results or paths["results"]))
    args.panel = str(ds.resolve_path(args.panel or paths["panel"]))
    args.figures = str(ds.resolve_path(args.figures or paths["figures"]))
    os.makedirs(args.figures, exist_ok=True)

    numbers = pn.load(args.results)
    f = lambda n: os.path.join(args.figures, n)

    fig_heterogeneity(numbers, args.panel, f("fig1_heterogeneity.png"))
    print("  fig1_heterogeneity.png")
    cross_path = ds.resolve_path("results/cross_window.json")
    if not os.path.exists(cross_path):
        raise SystemExit(
            f"{cross_path} not found — run `python src/cross_window.py` first. "
            "fig2 is cross-window evidence and is built from that file, not "
            "from a single window's results.")
    with open(cross_path) as fh:
        cross = json.load(fh)
    fig_ablation(cross, f("fig2_ablation.png"))
    print(f"  fig2_ablation.png          ({len(cross['windows'])} windows)")
    fig_arms(numbers, f("fig3_arms.png"))
    print("  fig3_arms.png")
    fig_client_gain(numbers, args.results, f("fig4_client_gain.png"))
    print("  fig4_client_gain.png")

    written = list(DEFAULT_FIGURES)
    if args.all:
        fig_convergence(numbers, args.results, f("convergence.png"))
        print("  convergence.png            (optional)")
        written.append("convergence.png")
    if args.all or args.forecast:
        fig_forecast(numbers, args.panel, f("forecast_example.png"))
        print("  forecast_example.png       (optional)")
        written.append("forecast_example.png")

    present = [p for p in os.listdir(args.figures) if p.endswith(".png")]
    optional = sorted(p for p in present
                      if p in OPTIONAL_FIGURES and p not in written)
    stale = sorted(p for p in present
                   if p not in written and p not in OPTIONAL_FIGURES)
    if optional:
        print(f"\n  {len(optional)} optional figure(s) from an earlier --all run:")
        for p in optional:
            print(f"    {p}")
    if stale:
        print(f"\n  {len(stale)} file(s) in {args.figures}/ were not written by "
              f"this run and are stale:")
        for p in stale:
            print(f"    {p}")
        print("  They may be out of date.")


if __name__ == "__main__":
    main()
