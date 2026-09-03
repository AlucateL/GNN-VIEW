"""Figure 3 builder — Explain mode, ERA5.

Loads a dataset + model, runs inference, computes errors and attributions for one
node, and renders the paper figure in the tool's own visual language.

Four layouts are exported:

  make_figure_vrank()  — "version A". Rows are targets, columns are
      [error] [attribution] [attribution to top source]* [ranking], with the
      ranking drawn vertically (features on x, tilted labels).
  make_figure_split()  — "version split". All maps on one row, the rankings on
      the next; each ranking spans its own target's pair of maps. V2_RANK_HORIZ
      draws those rankings as horizontal bar panels instead of vertical ones.
  make_figure_line()   — "version line". The split layout on ONE line: per target
      [error] [attribution] [ranking], targets divided by the dotted rule —
      2 maps · barplot ¦ 2 maps · barplot. FIG_WIDTH wide, with V2_MAP_W :
      V2_RANK_W as the proportions inside it, and the same V2_RANK_HORIZ flag.
  make_figure_row()    — "version row". make_figure's panels and its horizontal
      ranking, all on one row: one [error] [attribution] [ranking] block per
      target, blocks divided by the split layout's dotted rule.
      (make_figure() — the original stacked horizontal-ranking layout — is kept too.)

Usage from a notebook:

    import figure3_builder as f3
    f3.configure(PRED_TIME="2020-08-27T12:00:00", NODE_LATLON=(31.5, -93.0), ...)
    fig = f3.make_figure_vrank()
    f3.save(fig, f3.fig_stem("A"))

Every knob below is a module global; `configure()` is the only supported way to
change one, because it validates the name, keeps the font roles in sync with
FONT_S / FONT_XS, and reloads the model when the data selection changes.

Per-row node: a ROWS entry may carry `node=(lat, lon)` to override NODE_LATLON
for that row only — target node, attribution, map outline and (unless
CROP_CENTER pins it) the crop all follow that node.
"""

import ast
import json
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DATA_ROOT", str(ROOT / "backend" / "data"))
os.environ.setdefault("NEURAL_LAM_PATH", str(ROOT / "backend" / "neural-lam"))
if str(ROOT / "backend" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend" / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ── the figure's type family ─────────────────────────────────────────────────
# Named, and set before anything is drawn. Left to the default, matplotlib renders in
# DejaVu Sans, whose stems are heavier than what kaleido resolves for figure2_builder
# -- so the same #64748b out of colors.json read as two different greys across the
# plate. This is figure2_builder.SANS, in matplotlib's list form.
#
# Every entry must carry a real Bold: the panel letters and the self-variable ranking
# ticks below ask for one, and matplotlib answers a bold request from within the
# family it already matched. macOS ships HelveticaNeue as a .ttc collection whose
# Regular is the only face matplotlib registers, so naming it here returns the
# regular file for bold and the emphasis vanishes without a warning.
SANS = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = SANS
# Type 42 (TrueType) rather than matplotlib's default Type 3: Type 3 text is neither
# selectable nor searchable in the PDF, and is what most publishers reject outright.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from services import inference as inf  # noqa: E402
from services import integrated_gradients as IG  # noqa: E402

# TorchScript constants are baked on CPU; autograd needs everything on one device.
inf.DEVICE = "cpu"
IG.DEVICE = "cpu"
DEV = "cpu"

# ═══ Configuration ═══════════════════════════════════════════════════════════
# Everything adjustable lives between the two _CFG_START / _CFG_END markers;
# configure() rejects any name that is not declared here.
_CFG_START = set(globals())

# ── data / model / target ────────────────────────────────────────────────────
DATASET     = "aug2020.zarr"             # under backend/data/era5_datasets/
PRED_TIME   = "2020-08-27T12:00:00"      # timestep to predict
MODEL       = "graphcast"                # key into inf.PRESET_MODELS
NODE_LATLON = (31.5, -93.0)              # target node (nearest grid cell is used)

# ── attribution ──────────────────────────────────────────────────────────────
METHOD     = "ig"      # "ig" (paper figure, ~30 s per target) | "ixg" (~1 s, same story)
IG_STEPS   = 16
INPUT_STEP = "t-6h"    # "t-6h" | "t-12h" — which input timestep the attribution maps show

ATTR_SCALE = "max"     # colour scale of the attribution maps — see attr_vmax().
                       #   "max" (UI semantics): the largest |attr| outside the target node.
                       #     A single outlying cell then sets the scale for the whole panel
                       #     and compresses everything else to white, which makes a map with
                       #     one spike LOOK concentrated even when its background field is as
                       #     broad as a map without one.
                       #   "pct": the ATTR_PCT-th percentile of |attr| over the cells actually
                       #     drawn. The top tail saturates instead of setting the scale, so
                       #     two panels are comparable in shape. Clipped colorbars are marked
                       #     with the ≤ / ≥ their extremes.
                       #   Use "pct" to check whether an apparent difference in spatial extent
                       #   between two attribution panels is real or a dynamic-range artefact.
ATTR_PCT   = 99.0      # percentile used when ATTR_SCALE == "pct"

# ── what each row shows ──────────────────────────────────────────────────────
# target/source are (variable, level); level is None for surface fields.
# An optional node=(lat, lon) overrides NODE_LATLON for that row alone.
ROWS = [
    dict(target=("temperature", 925),              source=("temperature", 925)),
    dict(target=("total_precipitation_6hr", None), source=("total_precipitation_6hr", None)),
]

# ── map crop ─────────────────────────────────────────────────────────────────
CROP_CENTER = None            # (lat, lon) every crop is centred on.
                              #   None = centre each panel on its own row's node
CROP_HEIGHT = 26.0            # degrees of latitude at ZOOM = 1
ZOOM        = 0.8             # continuous; crop height = CROP_HEIGHT / ZOOM
CROP_ASPECT = 1.8             # reference width : height of a map panel
MAP_FILL    = 1.0             # 0 = keep CROP_ASPECT exactly; 1 = grow the maps vertically to
                              #   fill the row. Cells stay square either way — a taller panel
                              #   simply reveals more latitude at the same longitude span.

# ── figure geometry (inches) ─────────────────────────────────────────────────
FIG_WIDTH    = 5.5            # NeurIPS one-column
SHOW_TOP_SOURCE = False       # True  = 4 columns: error · attribution to the row's own source ·
                              #         attribution to the top-ranked source (when it differs) ·
                              #         ranking
                              # False = 3 columns, self-attribution only. Drops the top-ranked
                              #         map — the ranking already names it — and spends the
                              #         width on the remaining panels instead
WIDTH_RATIOS = [1, 1, 1, 1]   # relative column widths (truncated to the column count).
                              #   Ignored for any column whose width MAP_W / RANK_W pins
                              #   explicitly (make_figure only)

# Explicit panel sizes for make_figure, all in inches. None = derive them as before,
# from FIG_WIDTH / WIDTH_RATIOS / CROP_ASPECT. Setting any width makes FIG_WIDTH a
# reference for the *unpinned* columns only: the figure's own width then follows from
# what the columns actually add up to, the way the split layout already works.
MAP_W  = None                 # width of one map panel. Applies to every column but the
                              #   last, which is the ranking
RANK_W = None                 # width of a ranking's BAR AREA. Its row-label gutter
                              #   (RANK_LABEL_W, or auto) is added on top, so the column
                              #   is wider than this by that much
ROW_H  = None                 # height of a row's panels — maps and ranking share it.
                              #   Overrides the CROP_ASPECT / MAP_FILL derivation and the
                              #   RANK_TOP_N * RANK_ROW_PITCH floor (a warning is printed
                              #   if the ranking rows no longer fit). With MAP_W set, this
                              #   *is* the map aspect; CROP_ASPECT then only fixes the
                              #   longitude span of the crop, and the panel's shape decides
                              #   how much latitude comes with it
RANK_TOP_N   = 5              # rows in the ranking panels. Each row needs ~RANK_ROW_PITCH
                              #   of height; the figure grows to fit, so raising this makes
                              #   the figure taller rather than making labels collide.
FONT_S, FONT_XS = 7.0, 6.0
MARGIN   = dict(L=.03, R=.09, T=.34, B=.13)   # figure margins (T fits a 2-line title; R keeps
                                              #   the ranking x-axis label inside the figure)
TITLE_H  = .26                                # height a 2-line panel title occupies
COLGAP, ROWGAP = .09, .38                     # gaps between panels
CB_H, CB_GAP   = .050, .040                   # colorbar thickness and its gap below the map.
                                              #   The bar is horizontal, under the map and exactly
                                              #   as wide as it, so min/max sit on the map's edges
CB_LABELS      = True                         # unit label centred under each colorbar, on the
                                              #   same line as its min/max values. Set False to
                                              #   reclaim that line and put units in the titles
CB_LAB_H       = .105                         # height of the value/label line under the bar
                                              #   (.095 is enough when CB_LABELS is False)
RANK_LABEL_W   = None                         # gutter for ranking row labels, in inches.
                                              #   None = size it automatically from the longest
                                              #   label actually drawn (recommended — the labels
                                              #   change with the aggregation toggles)
RANK_SPAN_BLANK = False                       # True lets a ranking grow into a trailing blank
                                              #   column; False keeps every ranking the same
                                              #   width and leaves the top-right corner free
RANK_SHARED_XLIM = True                       # one x-scale for all ranking panels, so bar
                                              #   lengths are comparable between rows
RANK_XMAX      = 29.0                         # hard upper bound of the ranking x-axis, in % of
                                              #   attribution share. None = size it automatically
                                              #   from the longest bar plus legend headroom

RANK_TIME_AGG  = False                        # mirrors the UI's time-aggregation toggle.
                                              #   False (UI "off"): one row per feature PER input
                                              #     step — "Temp.@925 t−6h" — bar coloured by step
                                              #   True  (UI "on"):  one row per feature, bar split
                                              #     into its t−12h / t−6h shares
RANK_LEVEL_AGG = False                        # UI's level-aggregation toggle. True sums a
                                              #   variable's levels into one row ("Geopot.")
RANK_ROW_PITCH = FONT_XS * 1.8 / 72           # inches per ranking row (font height + leading)
RANK_LEGEND_HEADROOM = 1.55                   # extra x-range so the in-axes legend has clear
                                              #   space to the right of the shortest bars
RANK_BELOW     = .24                          # inches under a ranking panel for its x-axis
                                              #   ticks and axis label. The legend sits INSIDE
                                              #   the axes (lower right), where the descending
                                              #   bars leave the corner empty — so it costs no
                                              #   vertical space and the panels stay large.

# Ordered letters (a, b, c, ...) for a paper caption to cite. Numbered by POSITION —
# left to right along the top row, then along the bottom — so the sequence a reader
# scans is the sequence the caption cites. That is NOT the order the panels are drawn
# in: the split layout draws each target's maps and ranking together, so a draw-order
# counter would letter the top row a, b, d, e.
#
# The eleven names below are the SAME in figure3_builder and figure2_builder — same
# values, same meanings — so one plate can be lettered from one set of settings. Only
# PANEL_LETTER_LOC's default differs, and only because each figure's own title
# geometry forces it; see there.
PANEL_LETTERS      = False    # off by default; the figure stands alone without them
PANEL_LETTER_SCOPE = "all"    # "all" letters every panel | "maps" only the map panels,
                              #   leaving the rankings unlettered. A skipped panel does
                              #   not consume a letter either, so the sequence has no gaps.
PANEL_LETTER_START = "a"      # first letter, so a multi-figure plate can continue from "c"
PANEL_LETTER_FMT   = "{}"     # "{}" -> a  ·  "({})" -> (a)  ·  "{})" -> a)
PANEL_LETTER_LOC   = "above"  # "above" = the title's own baseline, outside the panel;
                              #   "inside" = the panel's top corner, over the map or the
                              #   chart. THIS figure defaults to "above": its titles are
                              #   short and centred, so they leave that line free.
                              #   figure2_builder defaults to "inside", where the titles
                              #   run the full panel width and "above" would collide.
PANEL_LETTER_SIDE  = "left"   # "left" | "right" — which corner. Honoured by both LOCs;
                              #   "right" anchors the letter's RIGHT edge to the panel's
                              #   right edge, so it never overhangs.
PANEL_LETTER_PAD   = (0.0, 0.0)  # (dx, dy) in points: +x right, -x left, +y up, -y down.
                              #   The nudge on top of whatever LOC and SIDE resolve to —
                              #   use it to clear a legend or a tall first bar without
                              #   moving to the other corner.
                              #   ONE pair pads every letter the same. A LIST of pairs pads
                              #   them individually, in letter order a, b, c ..., and must
                              #   hold exactly one entry per letter — e.g.
                              #   [(0,0), (0,0), (-14,0), (0,0), (0,0), (-14,0)] to pull
                              #   the two ranking letters clear of their legends. A list of
                              #   the wrong length raises.
PANEL_LETTER_SIZE  = None     # point size; None = TITLE_FS, the panel title's size
PANEL_LETTER_BOLD  = True
PANEL_LETTER_WASH  = True     # white pad behind an "inside" letter, so it reads over a
                              #   heatmap or a bar rather than fighting with it
PANEL_LETTER_HEADROOM = 1.16  # extra range on a lettered RANKING, "inside" only, so the
                              #   letter sits in reserved space rather than on the bars.
                              #   Maps need none — a letter over the heatmap hides nothing.

ERROR_SCALE = "crop"          # "shared" = one max across all error panels (UI semantics —
                              #   lets the rows be compared) | "crop" | "global"
COASTLINE   = "50m"           # 10m | 50m | 110m
DISPLAY_DPI = 200             # on-screen resolution
SAVE_DPI    = 600             # raster fallback resolution (the PDF itself stays vector)
SAVE_TIGHT  = True            # crop the saved file to what was actually drawn. The panels
                              #   are placed by hand, so the reserved gutters are estimates:
                              #   this trims the slack they leave and keeps any text that
                              #   overhangs the canvas. False = save the nominal canvas,
                              #   exactly FIG_WIDTH wide, slack and clipping included
SAVE_PAD    = .02             # inches of white kept around the content when SAVE_TIGHT
SAVE_DIR    = ROOT / "notebooks"

# ── typography ───────────────────────────────────────────────────────────────
# One family throughout (SANS, set with the imports above) — only size and colour vary.
# Five roles, honoured by every renderer here. configure() resets all of them
# whenever FONT_S / FONT_XS changes, unless the role itself is set in the same call.
TITLE_FS   = FONT_S    # panel titles ("Target error", "Ranked input fields …")
AXLAB_FS   = FONT_XS   # axis labels: colorbar unit, ranking share axis
TICKNUM_FS = FONT_XS   # numbers: colorbar extremes, ranking value axis, and the
                       #   value printed at each bar's tip
TICKLAB_FS = FONT_XS   # named ticks: the ranking's feature labels
LEGEND_FS  = FONT_XS   # the t−12h / t−6h key inside a ranking
CHIP_FS    = FONT_XS   # the field-name chip inside a map
# The legend's colour patch, in units of LEGEND_FS — matplotlib sizes handles relative
# to the legend's own font, so the box tracks the text automatically. These two set the
# proportion: raise them for a chunkier swatch at the same type size.
LEGEND_HANDLE_L, LEGEND_HANDLE_H = .8, .7

# ── vertical ranking (version A) ─────────────────────────────────────────────
RANK_TICK_ROT = 45          # degrees. 45 splits a label's cost evenly between width
                            #   and height; 0 needs ~3x the width, 90 ~3x the height
RANK_YMAX     = 29.0        # y-axis top, in % of attribution share (None = auto)
RANK_YLAB_W   = .30         # MINIMUM left gutter of a ranking column, in inches: y tick
                            #   numbers, the axis label, and the leftward overhang
                            #   of the tilted x labels

# ── split layout (version split) ─────────────────────────────────────────────
V2_MAP_W    = 1.2    # heatmap width (in). The figure width follows from it:
                     #   V2_LEFT + n·(2·V2_MAP_W + V2_COLGAP) + (n−1)·V2_GROUPGAP
                     #   + right margin — 5.50 in for two targets at these defaults
V2_ROW1_H   = .70    # height of the map row (in) — with V2_MAP_W this *is* the map
                     #   aspect, so CROP_ASPECT is not consulted in this layout
V2_ROW2_H   = .70    # height of the ranking row (in), bars only: the tilted labels
                     #   and the axis label sit below it
V2_ROWSPACE = .0075  # clear space between the rows (in), measured from the bottom of
                     #   the row-1 colorbar labels to the top of the row-2 titles
V2_COLGAP   = .09    # gap between the two maps of one target (in)
V2_GROUPGAP = .28    # gap between the two targets (in) — it opens between the central
                     #   heatmaps and between the rankings at once, since a ranking is
                     #   pinned to its own target's pair of maps
V2_DIVIDER  = True   # dotted rule down the middle of that gap, marking the boundary
                     #   between the two conditions
V2_DIV_SCALE = 1.00  # vertical length of that rule as a fraction of its full span (the
                     #   top of the heatmaps down to the ranking's x axis), shrunk about
                     #   the midpoint of that span, which stays fixed
V2_LEFT     = .30    # MINIMUM left margin (in): the ranking's y tick numbers and axis
                     #   label. It grows on its own if the tilted tick labels reach
                     #   further left than that — see `left` in make_figure_split.
                     #   With V2_RANK_HORIZ the tilted labels are gone and the row-label
                     #   gutter sits INSIDE the block, so this is taken as written
V2_RANK_HORIZ = False  # orientation of the ranking in the split and line layouts.
                     #   False = draw_ranking_v: features on x, share up y, labels tilted
                     #     under the axis (the paper default)
                     #   True  = draw_ranking: horizontal bars, one row per feature with
                     #     its label in a left-hand gutter. The gutter is reserved inside
                     #     the ranking's own span, so the block keeps its width; below the
                     #     panel only RANK_BELOW is needed instead of the tilted-label
                     #     block, which makes the figure shorter. RANK_XMAX / xmax pin the
                     #     share axis here, exactly as RANK_YMAX does for the vertical one
V2_RANK_W   = 1.2    # width of the ranking's BAR AREA in the one-line layout. Its label /
                     #   axis gutter is added on top, so the ranking's column is wider than
                     #   this by that much. Unused by the split layout, where a ranking
                     #   spans its own target's pair of maps.
                     #   By default this and V2_MAP_W are read as a RATIO, not as inches:
                     #   six panels side by side rarely add up to FIG_WIDTH on their own,
                     #   so the line layout scales both to whatever is left of FIG_WIDTH
                     #   once the margins, the gaps and the label gutters are paid for —
                     #   2 : 1 here means each map is twice the ranking's bar area. See
                     #   V2_LINE_PIN_W to read them as inches instead
V2_LINE_PIN_W = False  # one-line layout only. False = FIG_WIDTH is the figure's width and
                     #   V2_MAP_W / V2_RANK_W set the proportions inside it (NeurIPS
                     #   one-column by default). True = they are inches, pinned, and the
                     #   figure's width follows from what the blocks add up to — the way
                     #   MAP_W / RANK_W work in make_figure
# The split layout is denser than version A: more ranking rows in a shorter panel, so
# the labels are turned upright and the small type is dropped a notch. Applied for the
# duration of make_figure_split() only — set to {} to render it with the values above.
# These are defaults, not pins: any name passed to configure() wins over its entry
# here, so configure(RANK_TOP_N=6) gives six rows in the split layout too.
SPLIT_OVERRIDES = dict(RANK_TOP_N=10, RANK_TICK_ROT=90,
                       TITLE_FS=7, AXLAB_FS=5, TICKNUM_FS=5, TICKLAB_FS=5)
# Same idea for the one-line layout, which is by far the denser of the two: six panels
# across one column, so each is about half the width of a split-layout map. Hence
# fewer ranking rows (the panel is only as tall as a map), smaller type, and no unit
# label under the colorbars — at this width its line does not fit between the bar's
# two extremes, and the units are already in the axis-free "(norm.)" convention the
# rest of the plate uses. The map and ranking titles shorten too; that is not a
# setting but part of the layout, see _make_figure_line().
# Applied for the duration of make_figure_line() only, and — like SPLIT_OVERRIDES —
# any name the notebook passed to configure() wins over its entry here.
LINE_OVERRIDES  = dict(RANK_TOP_N=5, RANK_TICK_ROT=90,
                       TITLE_FS=6, AXLAB_FS=5, TICKNUM_FS=5, TICKLAB_FS=5,
                       CB_LABELS=False, CB_LAB_H=.095, CHIP_FS=5)

_CFG_END = set(globals())

_CFG_CACHE: dict = {}      # config_keys() memo, keyed on this file's mtime


def config_keys() -> frozenset:
    """The names configure() accepts: every top-level assignment sitting between the
    _CFG_START and _CFG_END markers, read back out of this file.

    Deliberately not a set frozen at import. IPython's autoreload re-executes only
    the top-level statements whose source actually changed, so a `CONFIG_KEYS = ...`
    line that was not itself edited keeps its pre-edit value -- add a setting to the
    block above and the next configure() rejects a name that plainly exists in the
    file, until the kernel is restarted. Reading the markers back from the source
    costs one parse, is cached on the file's mtime, and cannot drift.
    """
    src = Path(__file__)
    mtime = src.stat().st_mtime
    if _CFG_CACHE.get("mtime") != mtime:
        names, inside = set(), False
        for node in ast.parse(src.read_text()).body:
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, ast.AnnAssign) else [])
            # `FONT_S, FONT_XS = 7.0, 6.0` is one Assign with a Tuple target, so the
            # targets have to be flattened -- eight settings are declared that way.
            flat = []
            for t in targets:
                flat.extend(t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])
            for t in flat:
                if not isinstance(t, ast.Name):
                    continue
                if t.id == "_CFG_START":
                    inside = True
                elif t.id == "_CFG_END":
                    inside = False
                elif inside:
                    names.add(t.id)
        _CFG_CACHE.update(mtime=mtime, keys=frozenset(names))
    return _CFG_CACHE["keys"]


def __getattr__(name):
    """f3.CONFIG_KEYS still reads as an attribute, but is answered by config_keys()
    so it can never be the stale snapshot that docstring describes."""
    if name == "CONFIG_KEYS":
        return config_keys()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_FONT_ROLES = ("TITLE_FS", "AXLAB_FS", "TICKNUM_FS", "TICKLAB_FS", "LEGEND_FS", "CHIP_FS")
_RELOAD_KEYS = {"MODEL", "DATASET", "COASTLINE"}
_EXPLICIT = set()      # names configure() has been given. A layout's own *_OVERRIDES
                       #   dict defers to these — see split_overrides()
_RESTORING = False     # True while overrides() sets or puts back values, so its own
                       #   configure() calls do not count as user intent


def configure(**kw):
    """Set configuration globals by name. Unknown names raise, so a typo in a
    notebook config cell fails loudly instead of being silently ignored.

    FONT_S / FONT_XS propagate to the six font roles that default to them, unless
    the role is given explicitly in the same call. Changing MODEL, DATASET or
    COASTLINE reloads what depends on them."""
    unknown = sorted(set(kw) - config_keys())
    if unknown:
        raise KeyError(f"unknown config name(s): {', '.join(unknown)}")
    named = set(kw)                       # before the FONT_S/FONT_XS fan-out below:
                                          #   only what the caller wrote counts as
                                          #   user intent for split_overrides()
    if "FONT_S" in kw or "FONT_XS" in kw:
        s, xs = kw.get("FONT_S", FONT_S), kw.get("FONT_XS", FONT_XS)
        for role in _FONT_ROLES:
            kw.setdefault(role, s if role == "TITLE_FS" else xs)
        kw.setdefault("RANK_ROW_PITCH", xs * 1.8 / 72)
    globals().update(kw)
    if not _RESTORING:
        _EXPLICIT.update(named)
    if _state and (_RELOAD_KEYS & set(kw)):
        load(force=True)


@contextmanager
def overrides(**kw):
    """configure() for the duration of a `with` block, then put everything back."""
    if not kw:
        yield
        return
    global _RESTORING
    before = {k: globals()[k] for k in kw}
    was, _RESTORING = _RESTORING, True
    try:
        configure(**kw)
        _RESTORING = was
        yield
    finally:
        _RESTORING = True
        configure(**before)
        _RESTORING = was


def _layout_overrides(defaults):
    """`defaults` minus everything the notebook set itself. Without this a layout
    would re-pin RANK_TOP_N, the tick rotation and the font sizes on every render,
    silently ignoring a configure() call that names them."""
    return {k: v for k, v in defaults.items() if k not in _EXPLICIT}


def split_overrides():
    """SPLIT_OVERRIDES, as defaults rather than pins — see _layout_overrides()."""
    return _layout_overrides(SPLIT_OVERRIDES)


def line_overrides():
    """LINE_OVERRIDES, as defaults rather than pins — see _layout_overrides()."""
    return _layout_overrides(LINE_OVERRIDES)


# ═══ UI palette — shared by every renderer ═══════════════════════════════════
PALETTE = json.load(open(ROOT / "frontend/nextjs/src/lib/colors.json"))

# error heatmap — HEATMAP_STOPS in lib/colors.ts (note the non-uniform stop positions)
_E = PALETTE["errorHeatmap"]
CMAP_ERR = LinearSegmentedColormap.from_list("uiError", [
    (0.000, _E["stop0"]), (0.010, _E["stop1"]), (0.333, _E["stop2"]),
    (0.667, _E["stop3"]), (0.833, _E["stop4"]), (1.000, _E["stop5"])])

# attribution — ATTR_STOPS in ExplainDrawer.tsx, teal -> white -> rose
_X = PALETTE["explain"]
CMAP_ATTR = LinearSegmentedColormap.from_list("uiAttr", [
    (0.00, _X["attrCold"]), (0.25, _X["attrColdMid"]), (0.50, "#ffffff"),
    (0.75, _X["attrHotMid"]), (1.00, _X["attrHot"])])


def css_color(s):
    """colors.json stores some chrome as CSS rgba() strings; matplotlib needs tuples."""
    if s.startswith("rgba"):
        r, g, b, a = [float(x) for x in s[s.index("(") + 1:s.index(")")].split(",")]
        return (r / 255, g / 255, b / 255, a)
    return s


UI_BG, UI_TXT, UI_MUT = (PALETTE["ui"]["background"], PALETTE["ui"]["text"],
                         PALETTE["ui"]["textMuted"])
PANEL_BG = css_color(PALETTE["ui"]["panelBackground"])   # rgba(255,255,255,.94)
UI_BORDER = css_color(PALETTE["ui"]["border"])           # rgba(0,0,0,.09)
BOX = dict(boxstyle="round,pad=0.28", facecolor=PANEL_BG, edgecolor=UI_BORDER, linewidth=.45)
# The ranking legend's outline. Not UI_BORDER: rgba(0,0,0,.09) over the panel's own
# near-white is a visibly dark hairline on paper, and the key is a caption, not a
# mark. This is the same grey figure2_builder gives its in-map key (LEGEND_BORDER
# there) — light enough to bound the card without being read as content.
LEGEND_BORDER = "#e8ebee"
C_T12H, C_T6H, C_ACCENT = _X["t12h"], _X["t6h"], "#000000"
COAST_COL = "#94a3b8"

# Colours are fixed per role; only the sizes above vary.
TITLE_FC   = UI_TXT
AXLAB_FC   = UI_MUT
TICKNUM_FC = UI_MUT
TICKLAB_FC = UI_TXT
LEGEND_FC  = UI_MUT
CHIP_FC    = UI_TXT

# VAR_SHORT, verbatim from ExplainRankingSidebar.tsx — the ranking rows here read
# exactly as they do in the UI.
VAR_SHORT = {"geopotential": "Geopot.", "specific_humidity": "Sp.Hum.",
             "temperature": "Temp.", "u_component_of_wind": "U-Wind",
             "v_component_of_wind": "V-Wind", "vertical_velocity": "Vert.Vel.",
             "2m_temperature": "2m Temp", "10m_u_component_of_wind": "10m U",
             "10m_v_component_of_wind": "10m V", "mean_sea_level_pressure": "MSLP",
             "total_precipitation_6hr": "Precip."}

STEP_OUT = "t+0h"                            # error map: the prediction itself


# ═══ Load model, data, grid, coastline ═══════════════════════════════════════
_state: dict = {}


def load(force=False):
    """Load the model, the dataset tensors, the grid and the coastline for the
    configured MODEL / DATASET / COASTLINE. Idempotent: repeated calls are free
    until one of those three changes."""
    key = (MODEL, DATASET, COASTLINE)
    if not force and _state.get("key") == key:
        return _state
    model_path = inf.PRESET_MODELS.get(MODEL, inf.MODEL_PATH)
    model = inf.load_torchscript(model_path, device=DEV)
    cache = inf.preload_era5_tensors(ROOT / "backend" / "data" / "era5_datasets" / DATASET,
                                     device=DEV, lazy=True)
    grid = inf.load_grid()
    lat, lon = grid["lat"], grid["lon"]

    # regular 1.5 deg grid, flattened as (lon, lat)
    nlon, nlat = 240, 121
    lon_1d = lon.reshape(nlon, nlat)[:, 0]
    lat_1d = lat.reshape(nlon, nlat)[0, :]

    with open(ROOT / "frontend/nextjs/public/geojson" / f"coastline-{COASTLINE}.geojson") as f:
        geo = json.load(f)
    coast = []
    for feat in geo["features"]:
        geom = feat["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        coast.extend(np.asarray(p, dtype=float) for p in parts)

    g = globals()
    g.update(
        model=model, cache=cache, MODEL_PATH=model_path,
        LAT=lat, LON=lon,
        ATM=cache["atm_vars"], SURF=cache["surf_vars"], LEVELS=cache["levels"],
        NL=len(cache["levels"]), NLON=nlon, NLAT=nlat,
        LON_1D=lon_1d, LAT_1D=lat_1d,
        LON_ORDER=np.argsort(lon_1d),        # lon comes as 0..178.5,-180..-1.5
        DLAT=float(abs(lat_1d[1] - lat_1d[0])),  # 1.5 deg grid spacing
        HALF=float(abs(lat_1d[1] - lat_1d[0])) / 2,
        COAST=coast,
        # Coastline bounding boxes, computed once — the per-draw filter is then
        # vectorised, which is what makes a redraw quick.
        COAST_BBOX=np.array([[c[:, 0].min(), c[:, 0].max(),
                              c[:, 1].min(), c[:, 1].max()] for c in coast]),
    )
    _state.clear()
    _state["key"] = key
    print(f"model {model_path.name} · {DATASET} · {len(coast)} coastline parts")
    return _state


def _ensure_loaded():
    if _state.get("key") != (MODEL, DATASET, COASTLINE):
        load()


def to_grid(field):
    """flat (N_grid,) -> (n_lat, n_lon) with ascending lon, polar cells masked like the UI."""
    g = field.reshape(NLON, NLAT)[LON_ORDER, :].T
    mask = np.broadcast_to(np.abs(LAT_1D)[:, None] > 85.05, g.shape)
    return np.ma.masked_array(g, mask)


def node_index(latlon):
    """Index of the grid cell nearest to (lat, lon)."""
    _ensure_loaded()
    lat, lon = latlon
    return int(np.argmin((LAT - lat) ** 2 + (((LON - lon + 180) % 360) - 180) ** 2))


def row_node(cfg):
    """(index, (lat, lon)) of a ROWS entry's node — its own `node` if it has one,
    the global NODE_LATLON otherwise."""
    latlon = cfg.get("node") or NODE_LATLON
    return node_index(latlon), latlon


def multi_node():
    """True when ROWS do not all share one node. The ranking titles then name the node
    they belong to, which is the only thing telling two otherwise identical rows apart."""
    return len({row_node(c)[0] for c in ROWS}) > 1


def node_label(idx):
    """'31.5°N 93.0°W' for the grid cell actually used."""
    la, lo = float(LAT[idx]), float(LON[idx])
    return f"{abs(la):.1f}°{'N' if la >= 0 else 'S'} {abs(lo):.1f}°{'E' if lo >= 0 else 'W'}"


# ═══ Compute ═════════════════════════════════════════════════════════════════
# error_field and attribution are memoised on the full selection (dataset, model,
# prediction time, node, method), so changing one target re-uses everything else.
_err_cache: dict = {}
_attr_cache: dict = {}


def _data_key():
    return (DATASET, MODEL, PRED_TIME)


def cidx(var, level=None):
    if var in SURF:
        return len(ATM) * NL + SURF.index(var)
    return ATM.index(var) * NL + LEVELS.index(int(level))


def key_of(var, level):
    return f"{var}@{level}" if level is not None else var


def error_field(var, level=None):
    """|prediction - truth| per node, normalised — exactly what the tool's error map shows."""
    _ensure_loaded()
    k = (_data_key(), key_of(var, level))
    if k not in _err_cache:
        _err_cache[k] = inf.compute_prediction_errors(
            model, cache, PRED_TIME, var, level, device=DEV)
    return _err_cache[k]


def attribution(var, level=None, node=None):
    """Attribution for one target at one node: per-source maps at both input steps
    plus the ranking. `node` is a grid index; None uses NODE_LATLON."""
    _ensure_loaded()
    node = node_index(NODE_LATLON) if node is None else node
    k = (_data_key(), key_of(var, level), node, METHOD, IG_STEPS)
    if k in _attr_cache:
        return _attr_cache[k]

    if METHOD == "ixg":
        full = IG.compute_input_x_gradients(model, cache, PRED_TIME, var, level,
                                            target_node=node, device=DEV)
    else:
        full = IG.compute_integrated_gradients(model, cache, PRED_TIME, var, level,
                                               target_node=node, steps=IG_STEPS,
                                               batch_size=4, device=DEV)
    # backend's ranking: sum |attribution| per feature over all nodes, split by input step
    rows, tot = [], 0.0
    for meta in full["sources"]:
        kk = meta["key"]
        s_p = float(np.abs(full["attributions_prev"][kk]).sum())
        s_pp = float(np.abs(full["attributions_prev_prev"][kk]).sum())
        rows.append(dict(feature=kk, variable=meta["variable"],
                         level=int(kk.split("@")[1]) if "@" in kk else None,
                         score=s_p + s_pp, t6h=s_p, t12h=s_pp))
        tot += s_p + s_pp
    for r in rows:
        r["pct"] = 100 * r["score"] / tot
    rows.sort(key=lambda r: -r["score"])

    _attr_cache[k] = dict(maps_t6h=full["attributions_prev"],
                          maps_t12h=full["attributions_prev_prev"],
                          ranking=rows)
    return _attr_cache[k]


def attr_map(target, source, node=None):
    """Attribution map for one (target, source) pair at the configured input timestep."""
    a = attribution(*target, node=node)
    return np.asarray(a["maps_t6h" if INPUT_STEP == "t-6h" else "maps_t12h"][key_of(*source)])


# ═══ Labels ══════════════════════════════════════════════════════════════════
def short(var, level=None):
    """Short field label: the UI's baseShortLabel() plus the pressure unit —
    'Temp.@925hPa'. The UI can leave hPa implicit because its level picker is on
    screen; a figure read on its own cannot. Every label in the figure — ranking
    rows, panel titles, map chips — goes through here, so they stay consistent."""
    b = VAR_SHORT.get(var, var)
    return f"{b}@{int(level)}hPa" if level is not None else b


def step_in():
    """The attribution maps' input step. Read at draw time, and written with a true
    minus so the figure matches the UI (INPUT_STEP uses an ASCII hyphen)."""
    return INPUT_STEP.replace("-", "−")


def badge_label(var, level=None, step=None):
    """Map chip: the field label plus the timestep it is taken at — 'Temp.@925hPa
    t−6h'. The step is worth spelling out because the attribution maps are inputs at
    t−6h/t−12h while the error map is the prediction at t+0h."""
    s = short(var, level)
    return f"{s} ({step})" if step else s


def short_key(feature):
    v, _, l = feature.partition("@")
    return short(v, int(l) if l else None)


def rank_label(feature, step=None):
    """UI shortLabel(): truncate at 18 chars. The input step is NOT appended — it is
    already encoded in the bar colour, so repeating it in the label is redundant."""
    s = short_key(feature)
    return s if len(s) <= 18 else s[:15] + "…"


def rank_label_width(rows):
    """Inches needed by the widest ranking row label. Measured empirically at 6 pt:
    ~0.0085 in per character per point of font size."""
    n = max((len(r["label"]) for r in rows), default=8)
    return n * TICKLAB_FS * 0.0085 + 0.05


# ═══ Panel renderers ═════════════════════════════════════════════════════════
def _crop_center(center=None):
    """Where a crop is centred: an explicit centre, else CROP_CENTER if it pins one,
    else the row's own node."""
    return center or CROP_CENTER or NODE_LATLON


def crop_for(w_in, h_in, center=None):
    """Crop bounds for a panel of the given size. The longitude span is fixed by
    CROP_HEIGHT * CROP_ASPECT / ZOOM; latitude follows the panel's shape, which is
    what keeps grid cells square on screen. A taller panel therefore shows more
    latitude rather than stretching the cells."""
    w_deg = CROP_HEIGHT * CROP_ASPECT / ZOOM
    h_deg = w_deg * (h_in / w_in)
    la, lo = _crop_center(center)
    return la - h_deg / 2, la + h_deg / 2, lo - w_deg / 2, lo + w_deg / 2


def crop_wh(w_in, h_in, center=None):
    """Crop bounds from a panel's own shape: CROP_HEIGHT / ZOOM degrees of latitude,
    and whatever longitude span keeps the cells square at that width : height."""
    h_deg = CROP_HEIGHT / ZOOM
    w_deg = h_deg * (w_in / h_in)
    la, lo = _crop_center(center)
    return la - h_deg / 2, la + h_deg / 2, lo - w_deg / 2, lo + w_deg / 2


def attr_vmax(a, bounds=None, node=None):
    """Symmetric colour limit for an attribution map. The target node is always excluded
    — its own attribution is an order of magnitude above every other cell and would flatten
    the panel — which is what the UI does too.

    ATTR_SCALE "max" returns the largest remaining |attr| anywhere on the grid.
    ATTR_SCALE "pct" returns the ATTR_PCT-th percentile of |attr| over the cells inside
    `bounds` only (the whole grid if bounds is None): the panel is scaled to the field the
    reader can actually see, and the top tail clips. Falls back to the max if the percentile
    is degenerate, so an all-zero crop can never produce a zero-width scale."""
    node = node_index(NODE_LATLON) if node is None else node
    m = np.abs(np.delete(a, node))
    if ATTR_SCALE != "pct":
        return float(m.max())
    lat, lon = np.delete(LAT, node), np.delete(LON, node)
    keep = np.abs(lat) <= 85.05                       # polar cells are masked out in to_grid
    if bounds is not None:
        lo_la, hi_la, lo_lo, hi_lo = bounds
        dlon = ((lon - (lo_lo + hi_lo) / 2 + 180) % 360) - 180   # wrap-safe, from crop centre
        keep &= ((lat >= lo_la) & (lat <= hi_la) &
                 (np.abs(dlon) <= (hi_lo - lo_lo) / 2))
    v = float(np.percentile(m[keep], ATTR_PCT)) if keep.any() else 0.0
    return v if v > 0 else float(m.max())


def error_vmax(t, shared=None, center=None):
    """Colour-scale max for an error panel. 'shared' mirrors the UI, where all
    single-component views at one timestep sit on the same scale — without it a
    near-perfect field looks as saturated as the worst one."""
    e = error_field(*t)
    if ERROR_SCALE == "global":
        return float(e[np.abs(LAT) <= 85.05].max())
    lo_la, hi_la, lo_lo, hi_lo = crop_for(CROP_ASPECT, 1.0, center)
    m = (LAT >= lo_la) & (LAT <= hi_la) & (LON >= lo_lo) & (LON <= hi_lo)
    v = float(e[m].max())
    return max(v, shared) if (ERROR_SCALE == "shared" and shared) else v


_PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def letter_pad(index: int) -> tuple[float, float]:
    """
    The (dx, dy) nudge for the letter at position `index`, in points.

    PANEL_LETTER_PAD is either ONE (dx, dy) applied to every letter, or a LIST of
    them -- one per letter, in letter order a, b, c ... -- for a figure where each
    corner is obstructed by something different. The two forms are told apart by
    what the sequence holds: two numbers is the shared pair, pairs is the per-letter
    list.

    A short list raises here rather than falling back to (0, 0): a letter silently
    left unpadded is the failure this whole feature exists to avoid. A long one is
    caught by check_letter_pad() once the figure's letter count is known.
    """
    pad = PANEL_LETTER_PAD
    if not len(pad):
        raise ValueError("PANEL_LETTER_PAD is empty; give one (dx, dy) for every "
                         "letter, or a single (dx, dy) for all of them")
    if not any(isinstance(e, (tuple, list, np.ndarray)) for e in pad):
        dx, dy = pad                                   # one pair, shared by every letter
        return float(dx), float(dy)
    if index >= len(pad):
        raise ValueError(
            f"PANEL_LETTER_PAD has {len(pad)} entries but this figure letters at "
            f"least {index + 1} panels. Give one (dx, dy) per letter, in letter "
            f"order, or a single (dx, dy) to pad them all the same.")
    dx, dy = pad[index]
    return float(dx), float(dy)


def check_letter_pad(n_letters: int) -> None:
    """Reject a per-letter PANEL_LETTER_PAD that does not match the figure exactly.

    Called once the figure knows how many letters it drew. Only a long list can
    reach here -- letter_pad() has already rejected a short one -- but an extra
    entry means the caller is padding a panel that does not exist, which is worth
    saying out loud rather than dropping.
    """
    pad = PANEL_LETTER_PAD
    if not PANEL_LETTERS or not len(pad):
        return
    if any(isinstance(e, (tuple, list, np.ndarray)) for e in pad) and len(pad) != n_letters:
        raise ValueError(
            f"PANEL_LETTER_PAD has {len(pad)} entries but this figure letters "
            f"{n_letters} panels. Give exactly one (dx, dy) per letter, in letter "
            f"order, or a single (dx, dy) to pad them all the same.")


_LETTER_INSET = 2.0        # points in from the panel edge, "inside"
_LETTER_LIFT  = 3.0        # points above it, "above" — the titles' own pad. Neither is a
                           #   setting: they are the minimum that stops a letter touching
                           #   what it sits against. PANEL_LETTER_PAD is the knob on top.


def panel_letter(ax, i, dx_pt=0.0, kind="map", headroom=""):
    """Ordered panel letter just above `ax`'s top-left corner, for a caption to cite.

    Sits on the title's own line, with the same 3 pt pad, so it costs no extra height.
    `i` is a position index supplied by the caller — only the caller knows where a
    panel physically sits — and the call is a no-op unless PANEL_LETTERS is on.

    dx_pt shifts the letter left of the axes, in points. A ranking panel's axes start
    after the y-axis gutter but its title is centred over the whole COLUMN, so a
    two-line title reaches the axes' left edge and a letter placed there lands on it.
    Passing the gutter width moves the letter out to the column edge, where it clears
    the title and lines up with the letters of the maps above it.

    kind is "map" or "rank", read against PANEL_LETTER_SCOPE. headroom is "" for a
    panel that can be drawn over freely (a heatmap), or the axis a ranking's bars grow
    along — "y" for the vertical rankings, "x" for the horizontal one — so an "inside"
    letter can reserve space above the tallest bar instead of landing on it.

    Returns 1 if a letter was drawn and 0 if not, so a caller running a counter can
    advance it by the return value and never leave a gap in the sequence."""
    if not PANEL_LETTERS or (PANEL_LETTER_SCOPE == "maps" and kind != "map"):
        return 0
    if PANEL_LETTER_SIDE not in ("left", "right"):
        raise ValueError(f'PANEL_LETTER_SIDE must be "left" or "right", '
                         f'got {PANEL_LETTER_SIDE!r}')
    right = PANEL_LETTER_SIDE == "right"
    pad_x, pad_y = letter_pad(i)
    size = PANEL_LETTER_SIZE or TITLE_FS
    start = _PANEL_LETTERS.index(PANEL_LETTER_START)

    if PANEL_LETTER_LOC == "above":
        xy, va, dy = ((1, 1) if right else (0, 1)), "bottom", _LETTER_LIFT
        dx, bbox = 0.0, None
    elif PANEL_LETTER_LOC == "inside":
        xy, va, dy = ((1, 1) if right else (0, 1)), "top", -_LETTER_INSET
        dx = -_LETTER_INSET if right else _LETTER_INSET
        bbox = (dict(facecolor="white", alpha=.86, edgecolor="none",
                     boxstyle="square,pad=.15") if PANEL_LETTER_WASH else None)
        # Reserve space rather than draw on the marks. Grows the far end of the axis the
        # letter sits against, so the data keeps its origin and only the limit moves.
        if headroom and PANEL_LETTER_HEADROOM != 1.0:
            get, set_ = ((ax.get_ylim, ax.set_ylim) if headroom == "y"
                         else (ax.get_xlim, ax.set_xlim))
            lo, hi = get()
            set_(lo, lo + (hi - lo) * PANEL_LETTER_HEADROOM)
    else:
        raise ValueError(f'PANEL_LETTER_LOC must be "above" or "inside", '
                         f'got {PANEL_LETTER_LOC!r}')

    # dx_pt is a layout fact, not a preference: a ranking panel's axes start after its
    # y-axis gutter, so a LEFT letter is pushed back out to the column edge. A RIGHT one
    # is already at the column edge and needs no correction.
    ax.annotate(PANEL_LETTER_FMT.format(_PANEL_LETTERS[start + i]),
                xy=xy, xycoords="axes fraction",
                xytext=(dx + pad_x + (0.0 if right else dx_pt), dy + pad_y),
                textcoords="offset points", ha="right" if right else "left", va=va,
                fontsize=size, fontweight="bold" if PANEL_LETTER_BOLD else "normal",
                # Above the in-axes legend, which also claims zorder 6 and would
                # otherwise swallow a letter placed in its corner. A hidden letter is
                # worse than an ugly one -- move it with PANEL_LETTER_PAD or SIDE.
                color=TITLE_FC, bbox=bbox, zorder=10)
    return 1


def draw_map(ax, field, cmap, vmin, vmax, title, cb_label, bounds, cb_fmt="%.2g",
             badge=None, cb_clip=False, node=None):
    lo_la, hi_la, lo_lo, hi_lo = bounds
    node = node_index(NODE_LATLON) if node is None else node
    lon_e = np.concatenate([LON_1D[LON_ORDER] - HALF, [LON_1D[LON_ORDER][-1] + HALF]])
    lat_e = np.concatenate([LAT_1D - HALF, [LAT_1D[-1] + HALF]])

    ax.set_facecolor(UI_BG)
    m = ax.pcolormesh(lon_e, lat_e, to_grid(field), cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="flat", rasterized=True)

    keep = ((COAST_BBOX[:, 1] > lo_lo) & (COAST_BBOX[:, 0] < hi_lo) &
            (COAST_BBOX[:, 3] > lo_la) & (COAST_BBOX[:, 2] < hi_la))
    ax.add_collection(LineCollection([COAST[i] for i in np.nonzero(keep)[0]],
                                     colors=COAST_COL, linewidths=.4, zorder=3))

    # selected node — black outline on exactly its heatmap cell
    ax.add_patch(Rectangle((LON[node] - HALF, LAT[node] - HALF), DLAT, DLAT,
                           fill=False, ec="black", lw=.7, zorder=5))

    ax.set_xlim(lo_lo, hi_lo); ax.set_ylim(lo_la, hi_la)
    ax.set_aspect("equal", adjustable="box")     # square cells, always
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#00000022"); s.set_linewidth(.5)
    if badge is not None:
        # Variable name as a UI-style chip inside the map, rather than in the title.
        ax.text(.04, .96, badge, transform=ax.transAxes, ha="left", va="top",
                fontsize=CHIP_FS, color=CHIP_FC, zorder=6, bbox=BOX)
    ax.set_title(title, fontsize=TITLE_FS, color=TITLE_FC, pad=3, linespacing=1.3)

    # Colorbar under the map, spanning exactly its width: the extremes then read off
    # the map's own left and right edges. The values are drawn as text rather than as
    # tick labels, which would be centred on the ticks and so overhang those edges.
    pos = ax.get_position()
    fig_h_in = ax.figure.get_size_inches()[1]
    cax = ax.figure.add_axes([pos.x0, pos.y0 - (CB_GAP + CB_H) / fig_h_in,
                              pos.width, CB_H / fig_h_in])
    cb = plt.colorbar(m, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(.3); cb.outline.set_edgecolor("#00000030")
    cb.set_ticks([])
    # A clipped scale has to say so: past the extremes the colour stops tracking the value.
    lo_p, hi_p = ("≤", "≥") if cb_clip else ("", "")
    ends = [(0.0, "left", lo_p + cb_fmt % vmin, TICKNUM_FS, TICKNUM_FC),
            (1.0, "right", hi_p + cb_fmt % vmax, TICKNUM_FS, TICKNUM_FC)]
    if CB_LABELS:
        ends.append((0.5, "center", cb_label, AXLAB_FS, AXLAB_FC))   # same line as the extremes
    for x, ha, txt, fs, fc in ends:
        cax.annotate(txt, xy=(x, 0), xycoords="axes fraction", xytext=(0, -2.5),
                     textcoords="offset points", ha=ha, va="top", fontsize=fs, color=fc)


def rank_rows(ranking):
    """Reproduce the UI's ranking rows, including its level- and time-aggregation
    toggles. Percentages always use the grand total, so they stay comparable."""
    entries = ranking
    if RANK_LEVEL_AGG:
        agg = {}
        for r in ranking:
            a = agg.setdefault(r["variable"], dict(feature=r["variable"], variable=r["variable"],
                                                   level=None, score=0., t6h=0., t12h=0.))
            a["score"] += r["score"]; a["t6h"] += r["t6h"]; a["t12h"] += r["t12h"]
        entries = sorted(agg.values(), key=lambda r: -r["score"])

    tot = sum(r["score"] for r in entries) or 1.0
    out = []
    if RANK_TIME_AGG:
        for r in entries:
            out.append(dict(label=rank_label(r["feature"]), feature=r["feature"],
                            t12=100 * r["t12h"] / tot, t6=100 * r["t6h"] / tot,
                            pct=100 * r["score"] / tot))
    else:
        # UI time-agg OFF: each feature becomes two rows, re-sorted by per-step score
        for r in entries:
            for step, key, col in (("t−12h", "t12h", C_T12H), ("t−6h", "t6h", C_T6H)):
                out.append(dict(label=rank_label(r["feature"], step), feature=r["feature"],
                                t12=100 * r[key] / tot if col == C_T12H else 0.0,
                                t6=100 * r[key] / tot if col == C_T6H else 0.0,
                                pct=100 * r[key] / tot))
        out.sort(key=lambda r: -r["pct"])
    return out


def _check_axis_top(rows, limit, axis):
    """The share axis is pinned by RANK_XMAX / RANK_YMAX so panels stay comparable.
    A bar past that limit would be drawn cut off, silently — say so instead."""
    if limit is None or not rows:
        return
    top = max(r["pct"] for r in rows)
    if top > limit:
        print(f"warning: top bar {top:.1f}% exceeds the pinned {axis} limit {limit:.1f}% "
              f"— raise it, or set it to None for an automatic scale")


def axis_arrows(ax, color="#00000055", size=2):
    """Arrowhead at the far tip of each visible spine — x to the right, y at the top.
    Filled triangle markers rather than annotations: they are sized in points, so they
    stay matched to the spine width whatever the panel's shape or the data limits."""
    for xy, marker in (((1, 0), ">"), ((0, 1), "^")):
        ax.plot(*xy, marker=marker, markersize=size, color=color, clip_on=False,
                transform=ax.transAxes, zorder=4)


def _style_legend(ax, loc):
    leg = ax.legend(loc=loc, fontsize=LEGEND_FS, frameon=True, fancybox=True,
                    ncol=1, handlelength=LEGEND_HANDLE_L, handleheight=LEGEND_HANDLE_H,
                    labelspacing=.12, handletextpad=.35, borderpad=.35, borderaxespad=.3,
                    labelcolor=LEGEND_FC)
    fr = leg.get_frame()
    fr.set_facecolor(PANEL_BG); fr.set_edgecolor(LEGEND_BORDER); fr.set_linewidth(.5)
    leg.set_zorder(6)
    return leg


def draw_ranking(ax, ranking, highlight=None, title="ranking", legend=True, xmax=None,
                 title_x=0.5, panel_h=1.0):
    """Ranking sidebar, matching the UI's rows, labels and colours."""
    rows = rank_rows(ranking)[:RANK_TOP_N][::-1]
    _check_axis_top(rows, RANK_XMAX, "x")
    y = np.arange(len(rows))
    t12 = np.array([r["t12"] for r in rows])
    t6 = np.array([r["t6"] for r in rows])

    ax.barh(y, t12, color=C_T12H, height=.6, zorder=2, label="t−12h")
    ax.barh(y, t6, left=t12, color=C_T6H, height=.6, zorder=2, label="t−6h")
    for i, r in enumerate(rows):
        ax.text(t12[i] + t6[i] + .3, i, f"{r['pct']:.1f}", va="center",
                fontsize=TICKNUM_FS, color=TICKNUM_FC)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=TICKLAB_FS, color=TICKLAB_FC)
    if highlight is not None:
        for i, r in enumerate(rows):
            if r["feature"] == highlight:
                ax.get_yticklabels()[i].set_color(C_ACCENT)
                ax.get_yticklabels()[i].set_fontweight("bold")
    # RANK_XMAX pins the axis, so every panel reads on the same scale and the tick
    # values are round. Without it: the longest bar plus extra right-hand headroom
    # when the legend is inside, so the legend never covers a bar or its value label.
    ax.set_xlim(0, RANK_XMAX if RANK_XMAX is not None else
                (xmax if xmax else float((t12 + t6).max()))
                * 1.30 * (RANK_LEGEND_HEADROOM if legend else 1.0))
    ax.set_ylim(-.6, len(rows) - .45)
    ax.set_facecolor("none")

    # real x and y axes
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(.5); ax.spines[side].set_edgecolor("#00000055")
    axis_arrows(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))
    ax.set_axisbelow(True)          # must come before grid(); an explicit zorder
    ax.grid(axis="x", color="#000000", alpha=.13, linewidth=.45)
    ax.tick_params(axis="x", labelsize=TICKNUM_FS, length=1.6, width=.4, pad=1.5,
                   colors=TICKNUM_FC)
    ax.tick_params(axis="y", length=0, pad=1.5)
    # Centred on the axes — i.e. on the bar area — not on the whole column.
    ax.set_xlabel("attribution share (%)", fontsize=AXLAB_FS, color=AXLAB_FC, labelpad=1.5)
    # The bar area sits to the right of the label gutter, so a title centred on the
    # axes would drift off the figure. title_x re-centres it over the whole column.
    ax.set_title(title, fontsize=TITLE_FS, color=TITLE_FC, pad=3, linespacing=1.3, x=title_x)
    if legend:
        # Inside the axes, lower right: rows are sorted descending, so that corner is
        # always empty. Costs no layout space.
        _style_legend(ax, "lower right")


def label_w(rows):
    """Width of the longest tick label, in inches (~.0085 in per character per point)."""
    return max((len(r["label"]) for r in rows), default=8) * TICKLAB_FS * .0085


def tick_block_h(rows, rot=None):
    """Inches a tilted x-label block needs under a vertical ranking: the longest
    label projected onto the vertical, plus one line for the axis label."""
    rot = RANK_TICK_ROT if rot is None else rot
    return label_w(rows) * np.sin(np.radians(rot)) + AXLAB_FS * 2.4 / 72


def tick_overhang_w(rows, axes_w, rot=None):
    """Inches the tilted labels reach LEFT of their axes. The labels are anchored at
    their right end under a tick, so the leftmost one runs past the axes corner by its
    own width projected onto the horizontal, less the first tick's inset. Reserving
    this is what keeps the leftmost label from being cut off the canvas."""
    rot = RANK_TICK_ROT if rot is None else rot
    n = max(len(rows), 1)
    first_tick = axes_w * .6 / (n + .2)          # xlim is (-.6, n-.4)
    return max(0.0, label_w(rows) * np.cos(np.radians(rot)) - first_tick)


def draw_ranking_v(ax, ranking, highlight=None, title="ranking", legend=True,
                   ymax=None, ylabel=True, title_x=.5):
    """draw_ranking rotated a quarter turn: features along x, share up y. Rows keep
    the UI's order, so the bars descend left to right and the top-right corner is
    free for the legend."""
    rows = rank_rows(ranking)[:RANK_TOP_N]
    _check_axis_top(rows, ymax, "y")
    x = np.arange(len(rows))
    t12 = np.array([r["t12"] for r in rows])
    t6 = np.array([r["t6"] for r in rows])
    top = t12 + t6

    ax.bar(x, t12, color=C_T12H, width=.62, zorder=2, label="t−12h")
    ax.bar(x, t6, bottom=t12, color=C_T6H, width=.62, zorder=2, label="t−6h")
    for i, r in enumerate(rows):
        ax.text(i, top[i] + .35, f"{r['pct']:.1f}", ha="center", va="bottom",
                fontsize=TICKNUM_FS, color=TICKNUM_FC)

    ax.set_xticks(x)
    # rotation_mode="anchor" pins each label's right end under its tick, so the
    # labels stay attached to their bars however long they get.
    ax.set_xticklabels([r["label"] for r in rows], fontsize=TICKLAB_FS, color=TICKLAB_FC,
                       rotation=RANK_TICK_ROT, ha="right", rotation_mode="anchor")
    if highlight is not None:
        for i, r in enumerate(rows):
            if r["feature"] == highlight:
                ax.get_xticklabels()[i].set_color(C_ACCENT)
                ax.get_xticklabels()[i].set_fontweight("bold")

    ax.set_xlim(-.6, len(rows) - .4)
    # No legend headroom here: the legend sits in the corner the descending bars
    # already vacate. The 1.30 is for the value labels above the tallest bar.
    ax.set_ylim(0, ymax if ymax else float(top.max()) * 1.30)
    ax.set_facecolor("none")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(.5); ax.spines[side].set_edgecolor("#00000055")
    axis_arrows(ax, size=2.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#000000", alpha=.13, linewidth=.45)
    ax.tick_params(axis="y", labelsize=TICKNUM_FS, length=1.6, width=.4, pad=1.5,
                   colors=TICKNUM_FC)
    ax.tick_params(axis="x", length=0, pad=1.0)
    if ylabel:
        ax.set_ylabel("attribution share (%)", fontsize=AXLAB_FS, color=AXLAB_FC, labelpad=1.5)
    else:
        ax.tick_params(axis="y", labelleft=False)   # panels share one y-scale
    ax.set_title(title, fontsize=TITLE_FS, color=TITLE_FC, pad=3, linespacing=1.3, x=title_x)
    if legend:
        _style_legend(ax, "upper right")


# ═══ Assemble ════════════════════════════════════════════════════════════════
def build_row(cfg, show_top=True, ncol=4):
    """-> list of panel specs for one row, padded to ncol with blanks."""
    tgt, src = cfg["target"], cfg["source"]
    node, _ = row_node(cfg)
    rank = attribution(*tgt, node=node)["ranking"]
    top = (rank[0]["variable"], rank[0]["level"])
    panels = [("error", tgt, None), ("attr", tgt, src)]
    if show_top and key_of(*top) != key_of(*src):
        panels.append(("attr", tgt, top))
    panels.append(("rank", tgt, src))
    return panels + [("blank", None, None)] * (ncol - len(panels))


def _shared_error_max():
    """One error scale across every row, when ERROR_SCALE says so. Each row's crop is
    measured around its own node, so a figure mixing nodes still gets one honest max."""
    if ERROR_SCALE != "shared":
        return None
    return max(error_vmax(c["target"], center=row_node(c)[1]) for c in ROWS)


def _col_widths(ncol, rank_w, avail):
    """Width of each column of one target's block, in inches.

    WIDTH_RATIOS is sized for the configured variant; the other one takes what it
    needs, falling back to equal columns if the list is too short. MAP_W / RANK_W
    then pin a column outright — the ranking is the last one, and RANK_W sizes its
    bar area, so its label gutter is added on top."""
    ratios = WIDTH_RATIOS[:ncol] if len(WIDTH_RATIOS) >= ncol else [1] * ncol
    share = [avail * r / sum(ratios) for r in ratios]
    return [(RANK_W + rank_w if j == ncol - 1 and RANK_W is not None else
             MAP_W if j < ncol - 1 and MAP_W is not None else share[j])
            for j in range(ncol)]


def _panel_h(map_w):
    """(map height, row height) in inches, for a row whose maps are map_w wide.

    A 1.8:1 map in a quarter of a 5.5 in figure is only ~0.4 in tall, but a ranking
    panel needs one text line per row. The row takes whichever is taller and the maps
    are centred inside it — so labels never collide, at the cost of some whitespace
    above and below the maps. ROW_H overrides the whole derivation, including that
    floor: it is a pin, so it warns rather than growing the figure."""
    rankh = RANK_TOP_N * RANK_ROW_PITCH              # bars only; axis + legend sit below
    if ROW_H is not None:
        if ROW_H < rankh:
            print(f"warning: ROW_H {ROW_H:.2f} in is under the {rankh:.2f} in that "
                  f"{RANK_TOP_N} ranking rows need at {TICKLAB_FS:g} pt "
                  f"— raise it, lower RANK_TOP_N, or set ROW_H = None")
        return ROW_H, ROW_H
    axh = map_w / CROP_ASPECT                        # map height at the reference aspect
    rowh = max(axh, rankh)
    # Maps grow into the leftover row height, but never below what the ranking rows
    # need — the ranking then uses this same height, so both line up exactly.
    axh = max(axh + MAP_FILL * (rowh - axh), rankh)
    return axh, max(rowh, axh)


def _rank_gutter(rrows):
    """Inches reserved left of every ranking's bars for its row labels: RANK_LABEL_W,
    or the widest label any panel in the figure actually draws."""
    return (RANK_LABEL_W if RANK_LABEL_W is not None
            else max(rank_label_width(v) for v in rrows))


def make_figure(show_top=None):
    """Original layout: rows are targets, the ranking is a horizontal bar sidebar.

    Axes are positioned by hand rather than by GridSpec: it is the only way to
    guarantee that a map panel's width/height equals CROP_ASPECT exactly, which is
    what keeps the grid cells square while still filling the panel.

    MAP_W, RANK_W and ROW_H pin those panel sizes directly instead, in inches; the
    figure then sizes itself around them. Cells stay square either way — the crop
    follows whatever shape the panel ends up with.

    show_top overrides SHOW_TOP_SOURCE for this call, so both variants can be built
    from one set of cached attributions without touching the config."""
    _ensure_loaded()
    top_src = SHOW_TOP_SOURCE if show_top is None else show_top
    ncol = 4 if top_src else 3
    rows = [build_row(c, top_src, ncol) for c in ROWS]
    nrow = len(rows)
    M = MARGIN
    shared = _shared_error_max()
    rrows = {i: rank_rows(attribution(*c["target"], node=row_node(c)[0])["ranking"])[:RANK_TOP_N]
             for i, c in enumerate(ROWS)}
    rank_xmax = (max(v[0]["pct"] for v in rrows.values()) if RANK_SHARED_XLIM else None)
    rank_w = _rank_gutter(rrows.values())

    # Column widths. The colorbars sit under their maps, so no horizontal gutter is
    # reserved and every map fills its column edge to edge. Any MAP_W / RANK_W pin
    # makes the figure width follow the columns instead of the other way round.
    colw = _col_widths(ncol, rank_w,
                       FIG_WIDTH - M["L"] - M["R"] - (ncol - 1) * COLGAP)
    fig_w = (FIG_WIDTH if MAP_W is None and RANK_W is None
             else M["L"] + sum(colw) + (ncol - 1) * COLGAP + M["R"])
    axw = list(colw)
    axh, rowh = _panel_h(axw[0])
    # Whatever sits under a ranking panel (ticks, axis label, legend) needs clearance
    # in the row gap and in the bottom margin. The gap must clear the ranking's
    # below-axis furniture AND the next row's two-line titles, which are drawn above
    # that row's axes.
    cb_below = CB_GAP + CB_H + CB_LAB_H     # bar + its value/label line, under every map
    rowgap = max(ROWGAP, RANK_BELOW + TITLE_H, cb_below + TITLE_H)
    botm = max(M["B"], RANK_BELOW, cb_below)
    fig_h = M["T"] + nrow * rowh + (nrow - 1) * rowgap + botm

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=DISPLAY_DPI)

    def place(i, j, w=None, dx=0.0, h=None):
        hh = rowh if h is None else h
        x = M["L"] + sum(colw[:j]) + j * COLGAP + dx
        y = fig_h - M["T"] - (i + 1) * rowh - i * rowgap + (rowh - hh) / 2
        return fig.add_axes([x / fig_w, y / fig_h,
                             (axw[j] if w is None else w) / fig_w, hh / fig_h])

    # Panel letters run in reading order over the panels that are actually drawn --
    # left to right along a row, then down -- so a blank padding column is skipped
    # rather than consuming a letter.
    p = 0
    for i, (cfg, panels) in enumerate(zip(ROWS, rows)):
        node, latlon = row_node(cfg)
        rank = attribution(*cfg["target"], node=node)["ranking"]
        for j, (kind, t, s) in enumerate(panels):
            if kind == "blank":
                continue
            if kind == "rank":
                # A ranking panel has no colorbar, so it can claim its own gutter —
                # and the trailing blank column too, which is otherwise dead space.
                w = colw[j] - rank_w
                if RANK_SPAN_BLANK and j + 1 < ncol and panels[j + 1][0] == "blank":
                    w += COLGAP + colw[j + 1]
                span = w + rank_w                       # full column the ranking occupies
                ax = place(i, j, w=w, dx=rank_w, h=axh)
                draw_ranking(ax, rank, xmax=rank_xmax,
                             title_x=(span / 2 - rank_w) / w, panel_h=axh,
                             highlight=key_of(*s), title="Ranked input fields")
                p += panel_letter(ax, p, dx_pt=-rank_w * 72, kind="rank", headroom="x")
                continue
            ax = place(i, j, h=axh)
            bounds = crop_for(axw[j], axh, latlon)
            if kind == "error":
                draw_map(ax, error_field(*t), CMAP_ERR, 0, error_vmax(t, shared, latlon),
                         "Target error", "|err| (norm.)", bounds,
                         badge=badge_label(*t, step=STEP_OUT), node=node)
            elif kind == "attr":
                a = attr_map(t, s, node=node)
                v = attr_vmax(a, bounds, node)
                draw_map(ax, a, CMAP_ATTR, -v, v, "Input attribution", "attr (norm.)",
                         bounds, cb_fmt="%+.1g", badge=badge_label(*s, step=step_in()),
                         cb_clip=ATTR_SCALE == "pct", node=node)
            p += panel_letter(ax, p)
    check_letter_pad(p)
    return fig


# ═══ Version row · make_figure's panels, on one single row ═══════════════════
# The same panels, renderers and sizing knobs as make_figure, laid out along one row
# instead of stacked: each target gets its own [error · attribution · ranking] block,
# blocks separated by V2_GROUPGAP and the split layout's dotted rule. Panel letters
# run left to right across the whole row -- a, b, c | d, e, f for two targets.
def make_figure_row(show_top=None):
    """make_figure on a single row, one block per target, blocks divided by a rule.

    Panel sizes come from the same knobs — MAP_W and RANK_W pin a column, ROW_H the
    panel height, and WIDTH_RATIOS splits what is left of FIG_WIDTH. The gap between
    targets and the rule down its middle are the split layout's: V2_GROUPGAP,
    V2_DIVIDER, V2_DIV_SCALE.

    show_top overrides SHOW_TOP_SOURCE for this call, as in make_figure. A block whose
    top-ranked source is its own still reserves the extra column, so every block keeps
    the same width and the dividers stay evenly spaced."""
    _ensure_loaded()
    top_src = SHOW_TOP_SOURCE if show_top is None else show_top
    ncol = 4 if top_src else 3
    blocks = [build_row(c, top_src, ncol) for c in ROWS]
    n = len(blocks)
    M = MARGIN
    shared = _shared_error_max()
    rrows = [rank_rows(attribution(*c["target"], node=row_node(c)[0])["ranking"])[:RANK_TOP_N]
             for c in ROWS]
    rank_xmax = (max(v[0]["pct"] for v in rrows) if RANK_SHARED_XLIM else None)
    rank_w = _rank_gutter(rrows)

    # One block's columns. FIG_WIDTH is shared out over n blocks, their inner column
    # gaps and the gaps between them; a MAP_W / RANK_W pin then sizes the figure from
    # the blocks instead.
    avail = (FIG_WIDTH - M["L"] - M["R"] - n * (ncol - 1) * COLGAP
             - (n - 1) * V2_GROUPGAP) / n
    colw = _col_widths(ncol, rank_w, avail)
    axw = list(colw)
    blockw = sum(colw) + (ncol - 1) * COLGAP
    fig_w = (FIG_WIDTH if MAP_W is None and RANK_W is None
             else M["L"] + n * blockw + (n - 1) * V2_GROUPGAP + M["R"])
    axh, rowh = _panel_h(axw[0])

    # One row, so nothing sits between two rows: the ranking's below-axis furniture and
    # the colorbars only have to clear the bottom margin.
    cb_below = CB_GAP + CB_H + CB_LAB_H     # bar + its value/label line, under every map
    botm = max(M["B"], RANK_BELOW, cb_below)
    fig_h = M["T"] + rowh + botm
    y_ax = botm + (rowh - axh) / 2          # bottom edge of every panel in the row

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=DISPLAY_DPI)

    def place(k, j, w=None, dx=0.0):
        x = M["L"] + k * (blockw + V2_GROUPGAP) + sum(colw[:j]) + j * COLGAP + dx
        return fig.add_axes([x / fig_w, y_ax / fig_h,
                             (axw[j] if w is None else w) / fig_w, axh / fig_h])

    p = 0
    for k, (cfg, panels) in enumerate(zip(ROWS, blocks)):
        node, latlon = row_node(cfg)
        rank = attribution(*cfg["target"], node=node)["ranking"]
        for j, (kind, t, s) in enumerate(panels):
            if kind == "blank":
                continue
            if kind == "rank":
                # A ranking panel has no colorbar, so it can claim its own gutter —
                # and the trailing blank column too, which is otherwise dead space.
                w = colw[j] - rank_w
                if RANK_SPAN_BLANK and j + 1 < ncol and panels[j + 1][0] == "blank":
                    w += COLGAP + colw[j + 1]
                span = w + rank_w                       # full column the ranking occupies
                ax = place(k, j, w=w, dx=rank_w)
                draw_ranking(ax, rank, xmax=rank_xmax,
                             title_x=(span / 2 - rank_w) / w, panel_h=axh,
                             highlight=key_of(*s), title="Ranked input fields")
                p += panel_letter(ax, p, dx_pt=-rank_w * 72, kind="rank", headroom="x")
                continue
            ax = place(k, j)
            bounds = crop_for(axw[j], axh, latlon)
            if kind == "error":
                draw_map(ax, error_field(*t), CMAP_ERR, 0, error_vmax(t, shared, latlon),
                         "Target error", "|err| (norm.)", bounds,
                         badge=badge_label(*t, step=STEP_OUT), node=node)
            elif kind == "attr":
                a = attr_map(t, s, node=node)
                v = attr_vmax(a, bounds, node)
                draw_map(ax, a, CMAP_ATTR, -v, v, "Input attribution", "attr (norm.)",
                         bounds, cb_fmt="%+.1g", badge=badge_label(*s, step=step_in()),
                         cb_clip=ATTR_SCALE == "pct", node=node)
            p += panel_letter(ax, p)
    check_letter_pad(p)

    if V2_DIVIDER:
        # One rule per boundary between targets, down the middle of the gap: the panels
        # either side of it answer the question for a different target. Full span: the
        # top of the panels down to the ranking's x axis, shrunk about its own centre
        # by V2_DIV_SCALE.
        y_mid, half = y_ax + axh / 2, axh / 2 * V2_DIV_SCALE
        for k in range(n - 1):
            xd = M["L"] + k * (blockw + V2_GROUPGAP) + blockw + V2_GROUPGAP / 2
            fig.add_artist(plt.Line2D([xd / fig_w] * 2,
                                      [(y_mid - half) / fig_h, (y_mid + half) / fig_h],
                                      transform=fig.transFigure, color=UI_MUT,
                                      linewidth=.5, linestyle=(0, (1, 2.2)), alpha=.6))
    return fig


# ═══ Version A · main layout, vertical ranking ═══════════════════════════════
# The ranking's features move onto the x-axis and their labels are tilted, so they
# cost height instead of a wide left-hand gutter. Panels, colours, chips and
# colorbars are make_figure's — only the ranking renderer and the column geometry
# change.
def make_figure_vrank(show_top=None):
    """Same grid as make_figure. Two simplifications follow from the rotation: every
    column is the same width, and the ranking no longer sets a minimum row height,
    so the maps keep CROP_ASPECT exactly and MAP_FILL has nothing left to do."""
    _ensure_loaded()
    top_src = SHOW_TOP_SOURCE if show_top is None else show_top
    ncol = 4 if top_src else 3
    rows = [build_row(c, top_src, ncol) for c in ROWS]
    nrow = len(rows)
    M = MARGIN
    shared = _shared_error_max()
    rrows = [rank_rows(attribution(*c["target"], node=row_node(c)[0])["ranking"])[:RANK_TOP_N]
             for c in ROWS]
    ymax = (RANK_YMAX if RANK_YMAX is not None else
            max(r[0]["pct"] for r in rrows) * 1.30 if RANK_SHARED_XLIM else None)

    colw = (FIG_WIDTH - M["L"] - M["R"] - (ncol - 1) * COLGAP) / ncol
    axh = colw / CROP_ASPECT
    # The gutter holds the y-axis furniture AND the leftward reach of the tilted
    # labels, whichever is larger — otherwise the first label runs into the panel
    # to its left, or off the canvas.
    ylab_w = max(RANK_YLAB_W, max(tick_overhang_w(r, colw - RANK_YLAB_W) for r in rrows))
    cb_below = CB_GAP + CB_H + CB_LAB_H
    rank_below = max(tick_block_h(r) for r in rrows)
    below = max(cb_below, rank_below)
    rowgap = max(ROWGAP, below + TITLE_H)
    fig_h = M["T"] + nrow * axh + (nrow - 1) * rowgap + max(M["B"], below)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h), facecolor="white", dpi=DISPLAY_DPI)

    def place(i, j, w=None, dx=0.0):
        x = M["L"] + j * (colw + COLGAP) + dx
        y = fig_h - M["T"] - (i + 1) * axh - i * rowgap
        return fig.add_axes([x / FIG_WIDTH, y / fig_h,
                             (colw if w is None else w) / FIG_WIDTH, axh / fig_h])

    p = 0                      # panel letters, in reading order -- see make_figure
    for i, (cfg, panels) in enumerate(zip(ROWS, rows)):
        node, latlon = row_node(cfg)
        for j, (kind, t, s) in enumerate(panels):
            if kind == "blank":
                continue
            if kind == "rank":
                w = colw - ylab_w
                # The node goes on the title's first line, which is otherwise short —
                # a third line would not fit in the TITLE_H the layout budgets.
                tag = f" · {node_label(node)}" if multi_node() else ""
                ax = place(i, j, w=w, dx=ylab_w)
                draw_ranking_v(ax, attribution(*t, node=node)["ranking"],
                               highlight=key_of(*s),
                               title=f"ranking{tag}\ntarget {short(*t)}", ymax=ymax,
                               title_x=(colw / 2 - ylab_w) / w)
                p += panel_letter(ax, p, dx_pt=-ylab_w * 72, kind="rank", headroom="y")
                continue
            ax = place(i, j)
            bounds = crop_for(colw, axh, latlon)
            if kind == "error":
                draw_map(ax, error_field(*t), CMAP_ERR, 0, error_vmax(t, shared, latlon),
                         "error", "|err| (norm.)", bounds,
                         badge=badge_label(*t, step=STEP_OUT), node=node)
            else:
                a = attr_map(t, s, node=node)
                v = attr_vmax(a, bounds, node)
                draw_map(ax, a, CMAP_ATTR, -v, v, "attribution", "attr (norm.)",
                         bounds, cb_fmt="%+.1g", badge=badge_label(*s, step=step_in()),
                         cb_clip=ATTR_SCALE == "pct", node=node)
            p += panel_letter(ax, p)
    check_letter_pad(p)

    nodes = " · ".join(dict.fromkeys(node_label(row_node(c)[0]) for c in ROWS))
    fig.text(M["L"] / FIG_WIDTH, 1 - .04 / fig_h,
             f"{PRED_TIME[:16]} UTC · {nodes} · {MODEL} · "
             f"{'IG R=' + str(IG_STEPS) if METHOD == 'ig' else 'IxG'} · "
             f"zoom {ZOOM:.2f}× · panel {colw / axh:.2f}:1",
             fontsize=FONT_XS, color=UI_MUT, va="top")
    return fig


# ═══ Version split · maps on one row, rankings on the next ═══════════════════
#   row 1:  [err · target1] [attr · target1] [err · target2] [attr · target2]
#   row 2:  [    ranking · target1     ]     [    ranking · target2     ]
# Each ranking spans the two maps of its own target, plus the gap between them.
# Geometry is driven by the V2_* panel sizes rather than by CROP_ASPECT: fixing a
# map's width and its row's height fixes its aspect, and the crop follows from that.
# V2_RANK_HORIZ turns row 2's rankings a quarter turn, into horizontal bar panels.
def make_figure_split():
    _ensure_loaded()
    with overrides(**split_overrides()):
        return _make_figure_split()


def _make_figure_split():
    M = MARGIN
    tgts = [c["target"] for c in ROWS]
    srcs = [c["source"] for c in ROWS]
    nodes = [row_node(c) for c in ROWS]
    n = len(ROWS)                                        # n targets -> 2n maps, n rankings
    pitch = V2_MAP_W + V2_COLGAP                         # map to map within one target
    rank_w = 2 * V2_MAP_W + V2_COLGAP                    # a ranking spans its own two maps
    group = rank_w + V2_GROUPGAP                         # target to target

    rrows = [rank_rows(attribution(*t, node=nd)["ranking"])[:RANK_TOP_N]
             for t, (nd, _) in zip(tgts, nodes)]
    ymax = (RANK_YMAX if RANK_YMAX is not None
            else max(r[0]["pct"] for r in rrows) * 1.30)
    rank_xmax = (max(r[0]["pct"] for r in rrows) if RANK_SHARED_XLIM else None)
    shared = _shared_error_max()

    # A horizontal ranking carries its row labels in a gutter at the left of its own
    # span, so the block's width is unchanged and only the bar area shrinks.
    horiz = V2_RANK_HORIZ
    rank_gut = _rank_gutter(rrows) if horiz else 0.0

    # The left margin holds the y-axis furniture AND the leftward reach of the tilted
    # labels, whichever is larger. Each label is anchored under its own tick and runs
    # past the axes corner — that overhang is what would otherwise fall off the canvas.
    # A horizontal ranking has neither, so V2_LEFT stands as written.
    left = V2_LEFT if horiz else max(V2_LEFT, max(tick_overhang_w(r, rank_w) for r in rrows))
    fig_w = left + n * rank_w + (n - 1) * V2_GROUPGAP + M["R"]

    cb_below = CB_GAP + CB_H + CB_LAB_H
    # Under a horizontal ranking sit only its x ticks and axis label; under a vertical
    # one, the whole block of tilted feature labels.
    rank_below = RANK_BELOW if horiz else max(tick_block_h(r) for r in rrows)
    fig_h = (M["T"] + V2_ROW1_H + cb_below + V2_ROWSPACE + TITLE_H + V2_ROW2_H
             + rank_below + M["B"])
    y_map = fig_h - M["T"] - V2_ROW1_H
    y_rank = M["B"] + rank_below

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=DISPLAY_DPI)

    def ax_at(x, y, w, h):
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    for k, (t, s, (node, latlon)) in enumerate(zip(tgts, srcs, nodes)):
        bounds = crop_wh(V2_MAP_W, V2_ROW1_H, latlon)
        x0 = left + k * group                            # left edge of this target's block
        ax = ax_at(x0, y_map, V2_MAP_W, V2_ROW1_H)
        draw_map(ax, error_field(*t), CMAP_ERR, 0, error_vmax(t, shared, latlon),
                 "Target error", "|err| (norm.)", bounds,
                 badge=badge_label(*t, step=STEP_OUT), node=node)
        panel_letter(ax, 2 * k)

        a = attr_map(t, s, node=node)
        v = attr_vmax(a, bounds, node)
        ax = ax_at(x0 + pitch, y_map, V2_MAP_W, V2_ROW1_H)
        draw_map(ax, a, CMAP_ATTR, -v, v, "Input attribution", "attr (norm.)", bounds,
                 cb_fmt="%+.1g", badge=badge_label(*s, step=step_in()),
                 cb_clip=ATTR_SCALE == "pct", node=node)
        panel_letter(ax, 2 * k + 1)

        # One y-scale for every ranking, so only the leftmost carries the numbers and
        # the axis label. Both keep a legend, though: across the divider they read as
        # separate panels, and the right one has no axis furniture to anchor it.
        # The one-line title already fills the panel, so a differing node goes on a
        # second line — TITLE_H budgets two, so this costs no extra height.
        tag = f"\n{node_label(node)}" if multi_node() else ""
        title = f"Ranked input fields | Target: {short(*t)}{tag}"
        if horiz:
            # The bars start after the row-label gutter, but the title is centred over
            # the ranking's full span — as in make_figure, title_x puts it back there.
            w = rank_w - rank_gut
            ax = ax_at(x0 + rank_gut, y_rank, w, V2_ROW2_H)
            draw_ranking(ax, attribution(*t, node=node)["ranking"], highlight=key_of(*s),
                         title=title, xmax=rank_xmax, panel_h=V2_ROW2_H,
                         title_x=(rank_w / 2 - rank_gut) / w)
            panel_letter(ax, 2 * n + k, dx_pt=-rank_gut * 72, kind="rank", headroom="x")
        else:
            ax = ax_at(x0, y_rank, rank_w, V2_ROW2_H)
            draw_ranking_v(ax, attribution(*t, node=node)["ranking"], highlight=key_of(*s),
                           title=title, ymax=ymax, ylabel=(k == 0))
            panel_letter(ax, 2 * n + k, kind="rank", headroom="y")  # top row used 2n

    # 2n maps plus one ranking per target, unless the rankings are unlettered.
    check_letter_pad(2 * n if PANEL_LETTER_SCOPE == "maps" else 3 * n)

    if V2_DIVIDER:
        # One rule per boundary between targets, down the middle of the gap: the
        # panels either side of it answer the question for a different target.
        # Full span: top of the heatmaps -> the ranking's x axis. V2_DIV_SCALE
        # shrinks it about that span's centre, which is held fixed.
        y_hi, y_lo = y_map + V2_ROW1_H, y_rank
        y_mid, half = (y_hi + y_lo) / 2, (y_hi - y_lo) / 2 * V2_DIV_SCALE
        y0, y1 = y_mid - half, y_mid + half
        for k in range(n - 1):
            xd = left + k * group + rank_w + V2_GROUPGAP / 2
            fig.add_artist(plt.Line2D([xd / fig_w] * 2, [y0 / fig_h, y1 / fig_h],
                                      transform=fig.transFigure, color=UI_MUT,
                                      linewidth=.5, linestyle=(0, (1, 2.2)), alpha=.6))
    return fig


# ═══ Version line · one single line: maps and ranking side by side ═══════════
#   [err · t1] [attr · t1] [ranking · t1] ¦ [err · t2] [attr · t2] [ranking · t2]
# The split layout unfolded onto one line: the same V2_* panel sizes and the same
# dotted rule between targets, but a target's ranking sits beside its two maps
# instead of under them. Every panel shares one height (V2_ROW1_H) and the crop
# follows from each map's own shape, as in the split layout.
#
# Width, though, is the other way round from the split layout: six panels on one line
# do not add up to a one-column figure on their own, so FIG_WIDTH is the pin here and
# V2_MAP_W : V2_RANK_W only sets the proportion of a map to a ranking's bar area
# inside it. V2_LINE_PIN_W = True restores the split layout's rule, reading both as
# inches and letting the figure's width follow.
def make_figure_line():
    """The split layout on a single line, one [map · map · ranking] block per target.

    Exactly FIG_WIDTH wide unless V2_LINE_PIN_W says otherwise; the panels share out
    what the margins, gaps and ranking gutters leave. V2_RANK_HORIZ picks the
    ranking's orientation here too. Panel letters run left to right across the whole
    line — a, b, c | d, e, f for two targets."""
    _ensure_loaded()
    with overrides(**line_overrides()):
        return _make_figure_line()


def _make_figure_line():
    M = MARGIN
    tgts = [c["target"] for c in ROWS]
    srcs = [c["source"] for c in ROWS]
    nodes = [row_node(c) for c in ROWS]
    n = len(ROWS)
    horiz = V2_RANK_HORIZ

    rrows = [rank_rows(attribution(*t, node=nd)["ranking"])[:RANK_TOP_N]
             for t, (nd, _) in zip(tgts, nodes)]
    ymax = (RANK_YMAX if RANK_YMAX is not None
            else max(r[0]["pct"] for r in rrows) * 1.30)
    rank_xmax = (max(r[0]["pct"] for r in rrows) if RANK_SHARED_XLIM else None)
    shared = _shared_error_max()

    # Panel widths. The gutter and the bar area depend on each other for a vertical
    # ranking — a wider panel spreads its ticks out, so its labels overhang less —
    # so the two are solved together, from the gutter the current widths imply.
    # Three passes: the dependence is weak and monotone, and it settles on the second.
    map_w, rank_bar = V2_MAP_W, V2_RANK_W
    fixed = M["L"] + M["R"] + n * 2 * V2_COLGAP + (n - 1) * V2_GROUPGAP
    for _ in range(3):
        # The ranking's gutter: its row labels when the bars are horizontal, else the
        # y-axis furniture and the leftward reach of the tilted feature labels. Either
        # way it sits inside the block, between the second map and the bar area — no
        # ranking here is at the figure's left edge, so nothing can fall off the canvas.
        rank_gut = (_rank_gutter(rrows) if horiz else
                    max(V2_LEFT, max(tick_overhang_w(r, rank_bar) for r in rrows)))
        if V2_LINE_PIN_W:                    # V2_MAP_W / V2_RANK_W are inches, not ratios
            break
        avail = FIG_WIDTH - fixed - n * rank_gut
        if avail <= 0:
            raise ValueError(
                f"the line layout's margins, gaps and label gutters already take "
                f"{fixed + n * rank_gut:.2f} in of the {FIG_WIDTH:.2f} in FIG_WIDTH, "
                f"leaving nothing for the panels — raise FIG_WIDTH, drop a target, or "
                f"shorten the ranking labels with RANK_LEVEL_AGG / a smaller TICKLAB_FS")
        scale = avail / (n * (2 * V2_MAP_W + V2_RANK_W))
        map_w, rank_bar = V2_MAP_W * scale, V2_RANK_W * scale

    blockw = 2 * map_w + 2 * V2_COLGAP + rank_gut + rank_bar
    fig_w = (M["L"] + n * blockw + (n - 1) * V2_GROUPGAP + M["R"] if V2_LINE_PIN_W
             else FIG_WIDTH)

    cb_below = CB_GAP + CB_H + CB_LAB_H
    rank_below = RANK_BELOW if horiz else max(tick_block_h(r) for r in rrows)
    # One line, so nothing sits between two rows: the colorbars and the ranking's
    # below-axis furniture only have to clear the bottom margin.
    botm = max(M["B"], cb_below, rank_below)
    fig_h = M["T"] + V2_ROW1_H + botm
    y_ax = botm                                  # bottom edge of every panel on the line

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=DISPLAY_DPI)

    def ax_at(x, w):
        return fig.add_axes([x / fig_w, y_ax / fig_h, w / fig_w, V2_ROW1_H / fig_h])

    p = 0                       # panel letters, in reading order — see make_figure
    for k, (t, s_, (node, latlon)) in enumerate(zip(tgts, srcs, nodes)):
        bounds = crop_wh(map_w, V2_ROW1_H, latlon)
        x0 = M["L"] + k * (blockw + V2_GROUPGAP)     # left edge of this target's block

        # Version A's one-word titles, not the split layout's two-word ones: a map here
        # is about half as wide, and "Input attribution" set over it would run into
        # the panel beside it whatever the type size.
        ax = ax_at(x0, map_w)
        draw_map(ax, error_field(*t), CMAP_ERR, 0, error_vmax(t, shared, latlon),
                 "error", "|err| (norm.)", bounds,
                 badge=badge_label(*t, step=STEP_OUT), node=node)
        p += panel_letter(ax, p)

        a = attr_map(t, s_, node=node)
        v = attr_vmax(a, bounds, node)
        ax = ax_at(x0 + map_w + V2_COLGAP, map_w)
        draw_map(ax, a, CMAP_ATTR, -v, v, "attribution", "attr (norm.)", bounds,
                 cb_fmt="%+.1g", badge=badge_label(*s_, step=step_in()),
                 cb_clip=ATTR_SCALE == "pct", node=node)
        p += panel_letter(ax, p)

        # Two lines, as in version A: the ranking's column is no wider than a map and
        # a half, so the target does not fit on the title's own line. A differing node
        # joins it there rather than claiming a third line TITLE_H does not budget.
        tag = f" · {node_label(node)}" if multi_node() else ""
        title = f"ranking\ntarget {short(*t)}{tag}"
        span = rank_gut + rank_bar                   # the ranking's full column
        xr = x0 + 2 * (map_w + V2_COLGAP)
        ax = ax_at(xr + rank_gut, rank_bar)
        if horiz:
            draw_ranking(ax, attribution(*t, node=node)["ranking"], highlight=key_of(*s_),
                         title=title, xmax=rank_xmax, panel_h=V2_ROW1_H,
                         title_x=(span / 2 - rank_gut) / rank_bar)
        else:
            # Every ranking keeps its own y numbers: they are not adjacent here, so a
            # shared scale read off a distant axis would be a guess.
            draw_ranking_v(ax, attribution(*t, node=node)["ranking"], highlight=key_of(*s_),
                           title=title, ymax=ymax,
                           title_x=(span / 2 - rank_gut) / rank_bar)
        p += panel_letter(ax, p, dx_pt=-rank_gut * 72, kind="rank",
                          headroom="x" if horiz else "y")
    check_letter_pad(p)

    if V2_DIVIDER:
        # One rule per boundary between targets, down the middle of the gap: the panels
        # either side of it answer the question for a different target. Full span: the
        # panels' own height, shrunk about its centre by V2_DIV_SCALE.
        y_mid, half = y_ax + V2_ROW1_H / 2, V2_ROW1_H / 2 * V2_DIV_SCALE
        for k in range(n - 1):
            xd = M["L"] + k * (blockw + V2_GROUPGAP) + blockw + V2_GROUPGAP / 2
            fig.add_artist(plt.Line2D([xd / fig_w] * 2,
                                      [(y_mid - half) / fig_h, (y_mid + half) / fig_h],
                                      transform=fig.transFigure, color=UI_MUT,
                                      linewidth=.5, linestyle=(0, (1, 2.2)), alpha=.6))
    return fig


# ═══ Save ════════════════════════════════════════════════════════════════════
def _slug(var, level=None):
    """Filename-safe field tag from the UI's short label: 'temp925', 'precip'."""
    b = re.sub(r"[^a-z0-9]", "", VAR_SHORT.get(var, var).lower())
    return f"{b}{int(level)}" if level is not None else b


def _node_slug(idx):
    la, lo = float(LAT[idx]), float(LON[idx])
    return f"{abs(la):.1f}{'N' if la >= 0 else 'S'}{abs(lo):.1f}{'E' if lo >= 0 else 'W'}"


def fig_stem(variant):
    """Self-describing file stem: layout, the target fields, the predicted timestep
    and the node(s) — 'figure3_A_temp925-precip_20200827T1200_31.5N93.0W'. Repeated
    nodes collapse, so the usual single-node figure names stay short."""
    _ensure_loaded()
    fields = "-".join(_slug(*c["target"]) for c in ROWS)
    when = re.sub(r"[^0-9T]", "", PRED_TIME[:16])
    nodes = "-".join(dict.fromkeys(_node_slug(row_node(c)[0]) for c in ROWS))
    return f"figure3_{variant}_{fields}_{when}_{nodes}"


def save(fig, stem):
    """Write <stem>.pdf (vector) and <stem>.png under SAVE_DIR. With SAVE_TIGHT the
    file is cropped to the drawn content: the band of unused space under the tilted
    tick labels goes, and anything reaching past the canvas edge is kept rather than
    cut. Note that savefig(frameon=False) would not do this — it only drops the
    background colour, leaving the same empty space, now transparent."""
    out = Path(SAVE_DIR) / f"{stem}.pdf"
    kw = dict(facecolor="white")
    if SAVE_TIGHT:
        kw.update(bbox_inches="tight", pad_inches=SAVE_PAD)
    fig.savefig(out, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=SAVE_DPI, **kw)
    w, h = fig.get_size_inches()
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    sw, sh = ((bb.width + 2 * SAVE_PAD, bb.height + 2 * SAVE_PAD) if SAVE_TIGHT else (w, h))
    print(f"saved {out.name} + .png — canvas {w:.2f} x {h:.2f} in, file {sw:.2f} x {sh:.2f} in")
    return out
