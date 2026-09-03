"""
Flexible plotting for the graph-latency benchmark.

Reads what `benchmark_graph_scaling.py` writes and turns it into figures, without re-running
anything. The benchmark is slow (a full sweep is ~40 minutes); exploring the results should
not be, so everything here works off the two CSVs.

    from bench_plots import load, panels, component_plot, derivatives

    d = load("backend/data/bench_totals")
    panels(d)                                   # total latency vs g, m, e
    panels(d, y="edges_s")                      # same, but only the edge requests
    component_plot(d)                           # every component of every real graph
    derivatives(d)                              # fitted seconds-per-element

TWO TABLES, TWO QUESTIONS
-------------------------
`d.graphs` has one row per graph: the sizes (g, m, e), the two totals, and the per-dimension
latency aggregates (`grid_s`, `mesh_s`, `edges_s`). Use it for whole-render questions.

`d.components` has one row per request: which component, how many elements it serialized, how
long it took, how many bytes. Use it when the question is about one kind of component. The
real graphs contribute 48 rows here spanning 42 to ~2M elements, which is far more scaling
evidence than the 4 rows they contribute to `graphs`.

THE SWEEP COLUMN MATTERS
------------------------
Synthetic rows carry `sweeps`, naming which dimension was varied while the other two were
pinned. Fitting a slope against g over *all* synthetic rows mixes in points where g was held
constant and something else moved, which is meaningless. Every helper here filters on it, and
you should too when writing your own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import (FuncFormatter, LogFormatterSciNotation, LogLocator,
                               MaxNLocator, NullFormatter, ScalarFormatter)

# Columns that can sensibly go on an axis, with display labels.
AXES = {
    "g": "grid points",
    "m": "mesh nodes",
    "e": "total edges",
    "elements": "total elements (g + m + e)",
    "total_bytes": "total payload (bytes)",
    "total_seq_s": "total latency, sequential sum (s)",
    "total_parallel_s": "total latency, parallel wall clock (s)",
    "grid_s": "grid-point requests only (s)",
    "mesh_s": "mesh-node requests only (s)",
    "edges_s": "edge requests only (s)",
    "latency_s": "request latency (s)",
    "bytes_json": "payload (bytes, uncompressed)",
    "bytes_wire": "payload (bytes, gzipped)",
}

# Which dimension drives which component aggregate. Used by `panels` to pair each x axis with
# the part of the render that x actually causes.
COMPONENT_FOR = {"g": "grid_s", "m": "mesh_s", "e": "edges_s"}

# What each dimension is called when it is written out in a panel title.
ELEMENT_NAME = {"g": "grid points", "m": "mesh nodes", "e": "edges"}


# ---------------------------------------------------------------------------------------
# Figure style
#
# The target is a single-column NeurIPS figure: 5.5 inches is the full text width, and the
# figure is reproduced at that size with no further scaling. Every size below is therefore a
# *final* size on the page, which is why nothing here is smaller than 6pt -- a 5pt tick label
# survives the screen and dies in print.
# ---------------------------------------------------------------------------------------
FIG_WIDTH = 5.5                 # inches; NeurIPS \textwidth

FS_TICK = 6.5                   # tick labels and the x-axis power-of-ten offset
FS_LABEL = 7.0                  # axis labels
FS_TITLE = 7.5                  # panel titles
FS_LEGEND = 7.0                 # shared legend
FS_NOTE = 6.0                   # in-axes fit statistics

# Shipped graphs are coloured by family, with shades inside each family, so the eye sorts
# them into "weather" and "ocean" before it reads a single label. Green and red rather than
# anything in the blue range: blue is the measured-latency series in the first three panels,
# and a blue dot in panel 4 would read as the same thing.
REAL_FAMILIES = (
    ("weather", "#15803d", ("graphcast", "graph-efm", "efm", "era5", "weather", "icon",
                            "aifs", "atmos")),
    ("ocean", "#b91c1c", ("seacast", "njord", "ocean", "baltic", "sea", "nemo")),
)
FAMILY_FALLBACK = "#525252"

# Legend text for the shipped graphs. The benchmark keys them by the short name it ran under;
# a figure in a paper has to say what kind of mesh each one is, which is too long to sit in a
# CSV column and too important to leave to the caption.
DISPLAY_NAMES = {
    "GraphCast multiscale": "GraphCast-style (flat icosahedral)",
    "GraphCast hierarchical": "Graph-EFM (hierarchical icosahedral)",
    "SeaCast ocean": "SeaCast-style (hierarchical icosahedral-masked)",
    "Njord cluster": "Njord (hierarchical cluster-based)",
}


def display_name(name: str) -> str:
    """Legend text for a benchmarked graph; unmapped names are shown as they were recorded."""
    return DISPLAY_NAMES.get(str(name), str(name))


def _family(name: str) -> tuple[str, str]:
    """(family, base colour) for a graph name; unrecognised names get their own grey family."""
    low = str(name).lower()
    for fam, base, keys in REAL_FAMILIES:
        if any(k in low for k in keys):
            return fam, base
    return "other", FAMILY_FALLBACK


def _shades(base: str, n: int) -> list[str]:
    """`n` tints of `base`, full strength first, blending toward white but never past 50%."""
    rgb = np.asarray(to_rgb(base))
    if n <= 1:
        return [to_hex(rgb)]
    return [to_hex(rgb + (1.0 - rgb) * f) for f in np.linspace(0.0, 0.5, n)]


def real_colours(df: pd.DataFrame, name_col: str = "graph") -> dict[str, str]:
    """
    Graph name -> colour, one hue per family and a distinct shade per graph inside it.

    Built from the rows actually present rather than from a fixed table, so adding a graph to
    the benchmark re-spaces the shades instead of falling off the end of a palette.
    """
    names = list(dict.fromkeys(df[name_col].astype(str)))
    groups: dict[str, list[str]] = {}
    for nm in names:
        groups.setdefault(_family(nm)[0], []).append(nm)
    out = {}
    for fam, members in groups.items():
        base = next((b for f, b, _ in REAL_FAMILIES if f == fam), FAMILY_FALLBACK)
        for nm, col in zip(members, _shades(base, len(members))):
            out[nm] = col
    return out


def _xlabel(ax, text):
    """
    Axis label pushed to the left end, because the right end belongs to the exponent.

    A centred label and the offset text are both drawn on the same line below the ticks, and
    in a panel an inch wide they overlap every time. Left-aligning the label gives each of
    them its own end of the axis.
    """
    ax.set_xlabel(text, fontsize=FS_LABEL, ha="left", x=0.0)


def _despine(ax):
    """Top and right spines off; the two that carry ticks stay, softened."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#444")
    ax.tick_params(labelsize=FS_TICK, width=0.7, length=2.5, color="#444")


def _si(v: float) -> str:
    """Compact element count for titles: 8k, 300k, 3.5M."""
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _log_ticks(axis):
    """1-2-5 decade ticks written as plain numbers -- 0.5, 1, 2, 5, 20 rather than 10^0.3."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(NullFormatter())


def _sci_x(ax, log: bool = False):
    """
    Powers of ten on the x axis, one shared exponent parked at the right-hand end.

    Element counts run from 10^4 to 10^6 across the panels; written out in full they are both
    unreadable at 6.5pt and different lengths in every panel, which is what makes four panels
    look like four unrelated figures. A single "x10^5" at the right end of each axis reads at
    a glance and puts every panel on the same visual footing.
    """
    if log:
        ax.xaxis.set_major_locator(LogLocator(numticks=5))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(NullFormatter())
        return
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))            # always factor out an exponent, never inline zeros
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    off = ax.xaxis.get_offset_text()
    off.set_fontsize(FS_TICK)
    off.set_ha("right")                    # sits past the last tick, at the axis end


class BenchData:
    """The two tables plus convenience views. `graphs` and `components` are plain DataFrames."""

    def __init__(self, graphs: pd.DataFrame, components: pd.DataFrame, source: Path):
        self.graphs = graphs
        self.components = components
        self.source = source

    @property
    def real(self) -> pd.DataFrame:
        """Shipped graphs only."""
        return self.graphs[self.graphs["set"] == "real"]

    @property
    def synthetic(self) -> pd.DataFrame:
        """Generated sweep points only."""
        return self.graphs[self.graphs["set"] == "synthetic"]

    def sweep(self, dim: str) -> pd.DataFrame:
        """
        The synthetic points where `dim` was the dimension being varied, sorted by it.

        Filtering on this is what makes a slope meaningful -- see the module docstring.
        """
        s = self.synthetic
        keep = s["sweeps"].fillna("").str.split("|").apply(lambda t: dim in t)
        return s[keep].sort_values(dim)

    def __repr__(self) -> str:
        return (f"BenchData({self.source}: {len(self.graphs)} graphs "
                f"[{len(self.real)} real, {len(self.synthetic)} synthetic], "
                f"{len(self.components)} components)")


def load(path: str | Path) -> BenchData:
    """
    Loads a benchmark output directory (the one passed as --out).

    Adds two derived columns that are convenient as axes and tedious to recompute:
    `elements` (g + m + e) and `total_bytes`.
    """
    path = Path(path)
    graphs = pd.read_csv(path / "graphs.csv")
    components = pd.read_csv(path / "components.csv")
    graphs["elements"] = graphs["g"] + graphs["m"] + graphs["e"]
    graphs["total_bytes"] = (graphs["grid_bytes"] + graphs["mesh_bytes"]
                             + graphs["edges_bytes"])
    return BenchData(graphs, components, path)


def _fit(x, y, kind: str | None):
    """Returns (label, xline, yline, params) for a linear or power fit, or None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if kind is None or len(x) < 3:
        return None
    if kind == "linear":
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
        xs = np.linspace(x.min(), x.max(), 100)
        return (f"{a * 1e6:.2f} µs/element (R²={r2:.3f})", xs, a * xs + b,
                {"slope": a, "intercept": b, "r2": r2})
    if kind == "power":
        keep = (x > 0) & (y > 0)
        if keep.sum() < 3:
            return None      # a log fit needs positive values; zeros and NaNs are not data
        k, c = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
        pred = k * np.log(x[keep]) + c
        ly = np.log(y[keep])
        r2 = 1 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30)
        xs = np.geomspace(x[keep].min(), x[keep].max(), 100)
        return (f"k={k:.2f} (R²={r2:.3f})", xs, np.exp(c) * xs ** k,
                {"exponent": k, "r2": r2})
    raise ValueError(f"unknown fit {kind!r}; use None, 'linear' or 'power'")


def scaling_plot(df, x, y, ax=None, *, label=None, colour=None, marker="o", fit=None,
                 fit_label=None, logx=False, logy=False, annotate=None, filled=True,
                 line=True, ms=6):
    """
    The one primitive everything else is built from: y against x for a set of rows.

    Deliberately takes a DataFrame rather than a BenchData, so it works equally on `graphs`,
    on `components`, or on any slice you have filtered yourself.

    fit        None | "linear" (slope in seconds per element) | "power" (log-log exponent)
    fit_label  legend text for the fitted line. None keeps the default "fit: <slope> (R^2=...)";
               pass a plain name to put the statistics elsewhere, or "" to keep the line out of
               the legend entirely (which is what a shared legend wants for the 2nd panel on).
    annotate   column name whose value is written next to each point
    filled     False draws open markers, for showing a second series without hiding the first
    """
    ax = ax or plt.gca()
    # Rows with no measurement for either axis are dropped rather than drawn at zero -- see
    # the note in export_tables about runs that predate per-component recording.
    d = df.dropna(subset=[x, y]).sort_values(x)
    if d.empty:
        return ax
    style = dict(marker=marker, ms=ms, color=colour,
                 markerfacecolor=(colour if filled else "none"))
    if line:
        ax.plot(d[x], d[y], lw=1.6, label=label, **style)
    else:
        ax.plot(d[x], d[y], ls="none", label=label, **style)

    got = _fit(d[x], d[y], fit)
    if got:
        lbl, xs, ys, _ = got
        ax.plot(xs, ys, ls="--", lw=.9, alpha=.65, color=colour or "#444",
                label=(f"  fit: {lbl}" if fit_label is None else (fit_label or "_nolegend_")))
    if annotate:
        for _, r in d.iterrows():
            ax.annotate(str(r[annotate]), (r[x], r[y]), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _xlabel(ax, AXES.get(x, x))
    ax.set_ylabel(AXES.get(y, y), fontsize=FS_LABEL)
    ax.grid(True, which="major", alpha=.25, lw=.5)
    _sci_x(ax, log=logx)
    _despine(ax)
    return ax


def panels(data: BenchData, *, y="total_seq_s", xs=("g", "m", "e"), show_component=True,
           show_real=True, show_synthetic=True, fit="linear", logx=False, logy=False,
           figsize=None):
    """
    One panel per dimension, shared y axis: latency against g, m and e.

    Each panel filters the synthetic points to the sweep where that dimension was the one
    being varied, so the slope drawn is a real partial derivative rather than a mixture.

    show_component adds the aggregate that this dimension actually drives (grid_s for g,
    mesh_s for m, edges_s for e) as a second, open-marker series -- so a panel shows both the
    whole-render cost and the part attributable to its own axis.

    Set show_synthetic=False for a figure of the shipped graphs alone.
    """
    n = len(xs)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5.6 * n, 5), sharey=True)
    axes = np.atleast_1d(axes)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, x in zip(axes, xs):
        if show_synthetic:
            sw = data.sweep(x)
            if len(sw):
                scaling_plot(sw, x, y, ax=ax, label=f"synthetic ({AXES.get(y, y)})",
                             colour=colours[0], fit=fit, logx=logx, logy=logy)
                if show_component and COMPONENT_FOR.get(x) and COMPONENT_FOR[x] != y:
                    scaling_plot(sw, x, COMPONENT_FOR[x], ax=ax, filled=False,
                                 colour=colours[1], marker="s", fit=fit,
                                 label=f"synthetic ({AXES[COMPONENT_FOR[x]]})")
        if show_real:
            for i, (_, r) in enumerate(data.real.iterrows()):
                ax.plot([r[x]], [r[y]], marker="*", ms=15, ls="none",
                        color=colours[(i + 2) % len(colours)], zorder=5,
                        label=r["graph"] if x == xs[0] else None)
                if show_component and COMPONENT_FOR.get(x) and COMPONENT_FOR[x] != y:
                    ax.plot([r[x]], [r[COMPONENT_FOR[x]]], marker="*", ms=15, ls="none",
                            markerfacecolor="none",
                            color=colours[(i + 2) % len(colours)], zorder=5)
        ax.set_title(f"vs {AXES.get(x, x)}", fontsize=10)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(AXES.get(y, y), fontsize=FS_LABEL)
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=True)      # shared scale, but every panel keeps its numbers
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{AXES.get(y, y)} vs each dimension"
                 + ("   —  star = shipped graph, open = component only"
                    if show_real else ""), fontsize=11)
    fig.tight_layout()
    return fig, axes


def component_plot(data: BenchData, *, x="elements", y="latency_s", which="real",
                   group="graph", fit="power", logx=True, logy=True, ax=None, figsize=(8, 6)):
    """
    Every individual request as its own point: how long it took vs how much it returned.

    This is the densest real evidence in the dataset -- the shipped graphs alone give 48
    points across four and a half orders of magnitude, because each mesh level and each
    up/down subgraph is measured separately.

    group  "graph" colours by which graph the request came from; "component_kind" colours by
           nodes vs edges, which is the view that shows whether the two cost the same.
    """
    df = data.components
    if which in ("real", "synthetic"):
        df = df[df["set"] == which]
    if x == "elements":
        df = df[df["elements"] > 0]
    ax = ax or plt.subplots(figsize=figsize)[1]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (key, sub) in enumerate(df.groupby(group)):
        scaling_plot(sub, x, y, ax=ax, label=str(key), colour=colours[i % len(colours)],
                     line=False, logx=logx, logy=logy)
    got = _fit(df[x], df[y], fit)
    if got:
        lbl, xs_, ys_, _ = got
        ax.plot(xs_, ys_, "k--", lw=1, label=f"all points: {lbl}")
    ax.set_xlabel("elements returned by the request" if x == "elements" else AXES.get(x, x))
    ax.set_title(f"Per-component scaling ({which}, n={len(df)})", fontsize=10)
    ax.legend(fontsize=7)
    return ax


SERIES = {
    "total_seq_s": ("sequential sum", "#2563eb", "o"),
    "total_parallel_s": ("parallel wall clock", "#ea580c", "s"),
    "grid_s": ("grid-point requests", "#2563eb", "o"),
    "mesh_s": ("mesh-node requests", "#2563eb", "o"),
    "edges_s": ("edge requests", "#2563eb", "o"),
}

# What the two kinds of mark are called in the shared legend. The statistics that used to ride
# along in the fit's legend entry now sit inside the panel they belong to -- a legend shared by
# four panels cannot carry four different slopes.
LBL_POINTS = "measured server latency"
LBL_FIT = "linear fit"


def _clusters(values, log_gap=0.3):
    """
    Number of distinct groups in `values`, merging anything closer than `log_gap` decades.

    Counting *distinct* values is not a usable test of whether a fit is supported. The four
    shipped graphs have four distinct edge counts, but they are two pairs -- 153k/165k and
    3.39M/3.52M -- so a line through them is a two-point fit with an R^2 of 0.999 that means
    nothing. Grouping by ratio (0.3 dex ~ a factor of 2) counts what actually constrains a
    slope: how many separated places on the x axis have been sampled.
    """
    v = np.sort(np.asarray([x for x in values if x > 0], float))
    if v.size == 0:
        return 0
    return 1 + int((np.diff(np.log10(v)) > log_gap).sum())


def _panel_letters(fig, axes, letters="abcdefgh", x=-0.24):
    """
    (a), (b), (c) ... at the top-left of each panel, level with the top of its title.

    Two measurements rather than two guessed offsets. Vertically, the letter is aligned to the
    measured top of the title, because these titles run to two lines and a letter parked at a
    fixed height above the axes floats over a one-line title and collides with a three-line
    one. Horizontally it sits outside the axes, in the gutter the y tick labels use: at this
    width a title fills better than nine tenths of its panel, so there is no room for a letter
    inside the frame.
    """
    fig.canvas.draw()
    for ax, letter in zip(axes, letters):
        top = ax.title.get_window_extent().transformed(ax.transAxes.inverted()).y1
        ax.text(x, top, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_TITLE, fontweight="bold", clip_on=False)


def _rows(items, ncol: int) -> int:
    """Rows a legend of `items` occupies at `ncol` columns."""
    return -(-len(items) // max(ncol, 1))


def _fit_note(ax, notes):
    """
    Fit statistics parked in the corner an increasing trend leaves empty, one block per series.

    Each block is two lines (slope, then R^2), so blocks are stacked by their own height rather
    than by a single line height -- otherwise a second series overprints the first.
    """
    for i, (text, colour) in enumerate(reversed(notes)):
        ax.text(0.97, 0.03 + i * 0.20, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_NOTE, color=colour or "#444", zorder=6, linespacing=1.25,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=0.8))


def summary_figure(data: BenchData, *, ys=("total_seq_s",), real_x="e",
                   fit_real=None, real_scale="logy", figsize=(FIG_WIDTH, 2.7),
                   annotate_real=None, suptitle=None, letters=True):
    """
    The one figure: three derivative panels plus the shipped graphs on the right.

    Sized for print at 5.5 inches -- the NeurIPS full text width -- so it drops into a paper
    with no rescaling, and every font size in it is a final size on the page. Four panels in
    that width is tight, which is what the layout choices below are paying for: no per-panel
    legends (one shared legend under the figure), no suptitle by default (the caption says
    it), compact panel titles, and x axes that factor out a single power of ten instead of
    printing six-digit element counts.

    Panels 1-3 are synthetic only. Each plots whole-render latency against one dimension with
    the other two pinned, so the fitted slope is a partial derivative: seconds of server work
    per extra grid point, mesh node, or edge. They share a y axis, so the three slopes are
    directly comparable by eye -- and every panel keeps its own tick labels, because a shared
    axis read across a gap of an inch and a half is a shared axis in name only. The slope and
    R^2 of each fit are written into the panel rather than into the legend.

    `ys` is the list of series drawn on every panel; it defaults to the sequential sum alone,
    which is the server's own work and additive, so its slope is the derivative. Pass
    ys=("total_seq_s", "total_parallel_s") to add the parallel wall clock (what the browser
    waits, firing all requests at once). The two sit almost on top of each other on a
    single-worker backend, which is the finding rather than a mistake -- the requests are
    GIL-bound and queue regardless of how they are issued -- and at this figure width the
    overlap costs more in legibility than it buys.

    A component aggregate ("grid_s"/"mesh_s"/"edges_s") can be passed instead to isolate one
    component; the slope is the same, because within a sweep the other two contribute only a
    constant offset. That needs a run with per-component recording, and the panels say so if
    the column is empty.

    No real graphs on those three panels. A real graph pins none of the dimensions at the
    sweep baseline, so it cannot lie on a line drawn under that condition, and placing it
    there invites reading a distance that means nothing.

    Panel 4 is the real graphs alone: total render latency against edge count, edges being the
    dominant term (~85% of every render measured). Its own y axis, since whole renders on the
    large graphs sit well above the sweep's range. Each graph is one point in its own colour --
    blue shades for the weather graphs, amber shades for the ocean ones (see `real_colours`) --
    and the names live in the shared legend rather than in labels beside the points, which at
    this width would overlap each other. Green shades are the weather graphs and red shades
    the ocean ones -- deliberately not blue, which in the first three panels means "measured
    latency" and would read as that here.

    fit_real  None to draw points only; "linear" or "power" to fit. Off by default and skipped
              below 3 separated x clusters, because a line through two clusters of
              near-identical graphs is a two-point fit wearing a disguise.
    letters   (a)-(d) in the top-left corner of each panel, for referring to them from the
              caption. False for a figure that stands on its own.
    real_scale  axis scaling for panel 4. "logy" (default) keeps x linear -- so it carries the
              same factored power of ten as the other three panels -- and puts latency on a
              log axis, which is what separates the two members of a family: GraphCast
              multiscale and hierarchical differ by 0.9 s against a 22 s range, a single pixel
              apart on a linear axis and two clearly distinct points on a log one. "linear"
              for both axes linear, "loglog" once the shipped graphs span more than about two
              decades of edge count.
    """
    if real_scale not in ("linear", "logy", "loglog"):
        raise ValueError(f"unknown real_scale {real_scale!r}; "
                         "use 'linear', 'logy' or 'loglog'")
    dims = ("g", "m", "e")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 4)
    top = 0.855 if suptitle is None else 0.76
    # Provisional bottom; the real one is set once the legends have been drawn and measured.
    fig.subplots_adjust(left=0.085, right=0.995, top=top, bottom=0.35, wspace=0.62)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for a in axes[1:]:
        a.sharey(axes[0])
    ax_real = fig.add_subplot(gs[0, 3])

    missing, drawn = [], []
    for ax, x in zip(axes, dims):
        sw = data.sweep(x)
        notes = []
        for yk in ys:
            if sw.empty or yk not in sw or sw[yk].isna().all():
                missing.append(f"{yk} for the {x} sweep")
                continue
            lbl, colour, marker = SERIES.get(yk, (AXES.get(yk, yk), None, "o"))
            scaling_plot(sw, x, yk, ax=ax, colour=colour, marker=marker, ms=3.4,
                         label="_nolegend_", fit="linear", fit_label="")
            got = _fit(sw[x], sw[yk], "linear")
            if got:
                notes.append((got[0].replace(" (R", "\nR").rstrip(")"), colour))
            if yk not in drawn:
                drawn.append(yk)
        # The title names what is held fixed, one dimension per line and written out in full
        # ("8k mesh nodes", not "m=8000"). What is varied needs no saying twice: it is the x
        # axis, and the axis is labelled.
        held = ("\n".join(f"{_si(sw.iloc[0][v])} {ELEMENT_NAME[v]}" for v in dims if v != x)
                if not sw.empty else ELEMENT_NAME[x])
        ax.set_title(held, fontsize=FS_TITLE, linespacing=1.35)
        if notes:
            _fit_note(ax, notes)
        else:
            ax.text(.5, .5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FS_LABEL, color="#999")
        _xlabel(ax, AXES.get(x, x))
        ax.set_ylabel("latency (s)" if ax is axes[0] else "", fontsize=FS_LABEL)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _sci_x(ax)
        _despine(ax)
    # Shared y axis, but every panel keeps its numbers: the panels sit far enough apart that
    # reading panel 3 off panel 1's ticks is guesswork.
    for a in axes[1:]:
        a.tick_params(labelleft=True, labelsize=FS_TICK)

    real_all = data.real
    cols = real_colours(real_all)
    for yk in ys:
        if yk not in real_all:
            continue
        real = real_all.dropna(subset=[real_x, yk]).sort_values(real_x)
        if real.empty:
            continue
        _, _, marker = SERIES.get(yk, (AXES.get(yk, yk), None, "o"))
        # A thin white edge is what keeps the two members of a family readable: within a
        # family the graphs differ by under 8% in every dimension, so their points land on top
        # of one another and a bare fill would show three marks where there are four.
        for _, r in real.iterrows():
            ax_real.plot([r[real_x]], [r[yk]], ls="none", marker=marker, ms=4.2,
                         color=cols[str(r["graph"])], markeredgecolor="white",
                         markeredgewidth=0.5, label="_nolegend_", zorder=5)
        if fit_real and _clusters(real[real_x]) >= 3:
            got = _fit(real[real_x], real[yk], fit_real)
            if got:
                flbl, xs_, ys_, _ = got
                ax_real.plot(xs_, ys_, ls="--", lw=.9, alpha=.65, color="#444",
                             label="_nolegend_")
                _fit_note(ax_real, [(flbl.replace(" (R", "\nR").rstrip(")"), "#444")])
    if annotate_real:
        first = real_all.dropna(subset=[real_x, ys[0]])
        for _, r in first.iterrows():
            text = display_name(r[annotate_real]) if annotate_real == "graph" \
                else str(r[annotate_real])
            ax_real.annotate(text, (r[real_x], r[ys[0]]), fontsize=FS_NOTE,
                             xytext=(4, 3), textcoords="offset points",
                             color=cols[str(r["graph"])])
    n_clusters = _clusters(real_all[real_x])
    if fit_real and n_clusters < 3:
        print(f"  fit_real skipped: the real graphs occupy only {n_clusters} separated "
              f"{real_x} group(s) (need 3). {real_all[real_x].nunique()} distinct values, but "
              f"they cluster -- a slope through them would be a two-point fit in disguise. "
              f"Add graphs at more resolutions.")
    if real_scale != "linear":
        ax_real.set_yscale("log")
        _log_ticks(ax_real.yaxis)
    if real_scale == "loglog":
        ax_real.set_xscale("log")
    _xlabel(ax_real, AXES.get(real_x, real_x))
    ax_real.set_ylabel("total render latency (s)", fontsize=FS_LABEL)
    ax_real.set_title("GNN-VIEW\ndefault graphs", fontsize=FS_TITLE, linespacing=1.35)
    ax_real.grid(True, which="major", alpha=.25, lw=.5)
    if real_scale == "linear":
        ax_real.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _sci_x(ax_real, log=(real_scale == "loglog"))
    _despine(ax_real)

    # Two legends, stacked, rather than one grid: the mark entries are short and the graph
    # entries carry a mesh description, and mixing the two lengths in one grid leaves either a
    # ragged first column or a second column pushed off the figure.
    marks = []
    for yk in drawn:
        lbl, colour, marker = SERIES.get(yk, (AXES.get(yk, yk), None, "o"))
        name = LBL_POINTS if len(drawn) == 1 else f"{LBL_POINTS} ({lbl})"
        marks.append(Line2D([], [], color=colour, marker=marker, ms=3.4, lw=1.4, label=name))
    marks.append(Line2D([], [], color="#444", ls="--", lw=.9, alpha=.65, label=LBL_FIT))
    graphs = [Line2D([], [], color=c, marker="o", ms=4.2, ls="none", label=display_name(n),
                     markeredgecolor="white", markeredgewidth=0.5)
              for n, c in cols.items()]

    common = dict(loc="lower center", frameon=False, fontsize=FS_LEGEND, handlelength=1.6,
                  handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderaxespad=0.0)
    leg_graphs = fig.legend(handles=graphs, ncol=2, bbox_to_anchor=(0.5, 0.005), **common)

    # Both the second legend's position and the axes' bottom margin depend on how tall the
    # first one turned out, which depends on the label text. Rather than guess a row height
    # and be wrong for every label set but this one, draw once and measure.
    def _top(artist):
        return artist.get_window_extent().transformed(fig.transFigure.inverted()).y1

    fig.canvas.draw()
    # Two columns at most: a third entry of this length runs off both edges of a 5.5-inch page.
    leg_marks = fig.legend(handles=marks, ncol=min(len(marks), 2),
                           bbox_to_anchor=(0.5, _top(leg_graphs) + 0.012), **common)
    fig.canvas.draw()
    # 0.34 inches is what the tick labels and the left-aligned x label occupy below the axes.
    fig.subplots_adjust(bottom=min(0.6, _top(leg_marks) + 0.34 / figsize[1]))

    if letters:
        _panel_letters(fig, axes + [ax_real])

    if suptitle:
        fig.suptitle(suptitle, fontsize=FS_TITLE + 1)
    if missing:
        print(f"  note: missing data for {'; '.join(missing)}")
    return fig, (axes, ax_real)


def load_zoom(path: str | Path) -> pd.DataFrame:
    """The zoom.csv written by `benchmark_graph_scaling.py zoom`."""
    return pd.read_csv(Path(path) / "zoom.csv")


def zoom_plot(zoom_df: pd.DataFrame, *, graphs=None, figsize=(14, 5.5), annotate_zoom=True):
    """
    Crop latency: what a viewport costs, against how much it returns.

    Left panel is latency vs zoom level; right is latency vs elements actually returned, which
    is the causal axis. The dashed reference on the right is pure proportionality anchored at
    the largest crop -- points lifting above it at the small end are the fixed floor, the part
    of every request that does not shrink when you zoom in.

    That floor is why crop latency cannot be extrapolated from the full-view numbers: at full
    view everything is returned, so the floor is buried under serialization and invisible.
    """
    df = zoom_df if graphs is None else zoom_df[zoom_df["graph"].isin(graphs)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, sub) in enumerate(df.groupby("graph")):
        sub = sub.sort_values("zoom")
        c = colours[i % len(colours)]
        ax1.plot(sub["zoom"], sub["sequential_s"] * 1000, "o-", ms=5, color=c, label=name)
        s2 = sub.sort_values("elements_returned")
        ax2.plot(s2["elements_returned"], s2["sequential_s"] * 1000, "o-", ms=5, color=c,
                 label=name)
        if annotate_zoom:
            for _, r in s2.iterrows():
                ax2.annotate(f"z{int(r['zoom'])}", (r["elements_returned"],
                                                    r["sequential_s"] * 1000),
                             fontsize=6, xytext=(4, -7), textcoords="offset points")
        big = s2.iloc[-1]
        ref_x = np.geomspace(max(s2["elements_returned"].min(), 1),
                             big["elements_returned"], 50)
        ax2.plot(ref_x, big["sequential_s"] * 1000 * ref_x / big["elements_returned"],
                 ls="--", lw=.8, alpha=.5, color=c)

    ax1.set_xlabel("zoom level"); ax1.set_ylabel("total render latency (ms)")
    ax1.set_yscale("log"); ax1.set_title("vs zoom", fontsize=10)
    ax1.grid(True, which="both", alpha=.25, lw=.5); ax1.legend(fontsize=7)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("elements actually returned"); ax2.set_ylabel("total render latency (ms)")
    ax2.set_title("vs elements returned\n(dashed = pure proportionality; "
                  "lift at the left is the fixed floor)", fontsize=9)
    ax2.grid(True, which="both", alpha=.25, lw=.5)
    fig.suptitle("Crop latency by zoom level (Atlantic centre, 1200x800 viewport)",
                 fontsize=11)
    fig.tight_layout()
    return fig, (ax1, ax2)


def zoom_floor(zoom_df: pd.DataFrame, n_small=4) -> pd.DataFrame:
    """
    Fits `latency = floor + c * elements_returned` on each graph's smallest crops.

    Only the small crops, because that is where the floor is a large enough share of the total
    to be identifiable; at full view it is a percent or two and the fit would not see it.
    """
    rows = []
    for name, sub in zoom_df.groupby("graph"):
        s = sub.nsmallest(n_small, "elements_returned")
        if len(s) < 3:
            continue
        a, b = np.polyfit(s["elements_returned"], s["sequential_s"], 1)
        rows.append({"graph": name, "floor_ms": b * 1000, "us_per_element": a * 1e6,
                     "n_points": len(s),
                     "full_view_s": sub["sequential_s"].max(),
                     "zoom4_s": float(sub[sub["zoom"] == 4]["sequential_s"].iloc[0])
                     if (sub["zoom"] == 4).any() else float("nan")})
    return pd.DataFrame(rows).sort_values("full_view_s")


def derivatives(data: BenchData, ys=("total_seq_s", "total_parallel_s"),
                dims=("g", "m", "e")) -> pd.DataFrame:
    """
    Fitted seconds-per-element for each dimension: the derivative, as a table.

    `r2` is not decoration. A low value means the fit has not constrained that slope -- which
    is exactly what happens for mesh nodes against the *total*, because mesh requests are well
    under 1% of a render and the signal is swamped. In that case read the slope from the
    component aggregate (`ys=("mesh_s",)`) instead of the total.
    """
    rows = []
    for dim in dims:
        sw = data.sweep(dim)
        if len(sw) < 3:
            continue
        for y in ys:
            got = _fit(sw[dim], sw[y], "linear")
            if not got:
                continue
            _, _, _, prm = got
            rows.append({"dimension": dim, "y": y,
                         "us_per_element": prm["slope"] * 1e6,
                         "s_per_1000_added": prm["slope"] * 1000,
                         "r2": prm["r2"], "n_points": len(sw)})
    return pd.DataFrame(rows)


def additive_model(data: BenchData, y="total_seq_s") -> dict:
    """
    Non-negative least squares for y = a*g + b*m + c*e + d over every synthetic point.

    Non-negative because a negative cost per element is not physically meaningful; an
    unconstrained fit will happily produce one to absorb noise when a term is weak, and it
    then reads as though adding nodes makes the request faster.
    """
    from scipy.optimize import nnls

    s = data.synthetic
    A = np.column_stack([s["g"], s["m"], s["e"], np.ones(len(s))]).astype(float)
    coef, _ = nnls(A, s[y].to_numpy(float))
    pred = A @ coef
    obs = s[y].to_numpy(float)
    return {"us_per_grid_point": coef[0] * 1e6, "us_per_mesh_node": coef[1] * 1e6,
            "us_per_edge": coef[2] * 1e6, "const_s": coef[3],
            "r2": 1 - ((obs - pred) ** 2).sum() / max(((obs - obs.mean()) ** 2).sum(), 1e-30)}


def predict(model: dict, g: int, m: int, e: int) -> float:
    """Predicted latency in seconds from an `additive_model` result."""
    return (model["us_per_grid_point"] * g + model["us_per_mesh_node"] * m
            + model["us_per_edge"] * e) / 1e6 + model["const_s"]
