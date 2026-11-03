"""Turn run summaries into the README results table and a per-difficulty chart (light + dark PNGs)."""
from __future__ import annotations

import json
from pathlib import Path

from .evaluate import LEVELS

# Categorical palette, fixed order; validated for color-vision deficiencies on both surfaces.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]},
}


def load_runs(specs: list[str]) -> list[tuple[str, dict]]:
    """`label=path/to/run.summary.json` → [(label, summary)]."""
    runs = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"Expected label=path, got {spec!r}")
        runs.append((label, json.loads(Path(path).read_text())))
    return runs


def _levels(runs: list[tuple[str, dict]]) -> list[str]:
    """Difficulty levels present in the runs, in canonical order (Spider's or BIRD's)."""
    return [k for k in LEVELS if any(k in s["by_hardness"] for _, s in runs)]


def markdown_table(runs: list[tuple[str, dict]]) -> str:
    levels_order = _levels(runs)
    head = ("| Model | Max turns | Execution accuracy (95% CI) | First attempt | "
            + " | ".join(k.capitalize() for k in levels_order) + " | Cost |")
    lines = [head, "|" + "---|" * (5 + len(levels_order))]
    for label, s in runs:
        lo, hi = s["execution_accuracy_ci95"]
        levels = [f"{s['by_hardness'][k]['accuracy']:.0%}" if k in s["by_hardness"] else "–" for k in levels_order]
        cost = f"${s['total_cost_usd']:.2f}" if s.get("total_cost_usd") else "$0 (Colab)"
        lines.append(f"| {label} | {s['max_turns']} | **{s['execution_accuracy']:.1%}** ({lo:.1%}–{hi:.1%}) "
                     f"| {s['first_attempt_accuracy']:.1%} | " + " | ".join(levels) + f" | {cost} |")
    return "\n".join(lines)


def chart(runs: list[tuple[str, dict]], out_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(runs) > len(THEMES["light"]["series"]):
        raise ValueError("At most 6 runs per chart: split them into several charts.")
    levels_order = _levels(runs)
    groups = ["Overall", *[k.capitalize() for k in levels_order]]
    paths = []
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        width = 0.8 / len(runs)
        for i, (label, s) in enumerate(runs):
            values = [s["execution_accuracy"]] + [s["by_hardness"].get(k, {}).get("accuracy", 0) for k in levels_order]
            xs = [g + (i - (len(runs) - 1) / 2) * width for g in range(len(groups))]
            ax.bar(xs, [v * 100 for v in values], width, label=label, color=t["series"][i],
                   edgecolor=t["surface"], linewidth=1.5, zorder=3)
        ax.set_xticks(range(len(groups)), groups, color=t["ink"], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Execution accuracy (%)", color=t["muted"], fontsize=9)
        ax.tick_params(axis="y", colors=t["muted"], labelsize=8, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=min(len(runs), 3),
                           frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(t["ink"])
        fig.tight_layout()
        path = out_stem.with_name(f"{out_stem.name}-{mode}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=t["surface"])
        plt.close(fig)
        paths.append(path)
    return paths
