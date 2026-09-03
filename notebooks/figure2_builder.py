"""
Figure 2 — ocean graph defects. Everything the figure knows how to do.

One row of four panels: a cropped Inspect map beside the distribution it was
thresholded on, twice, split by a dotted rule.

    [ map A ] gap [ dist A ]   MIDGAP+rule   [ map B ] gap [ dist B ]

The left block thresholds mesh edges on LENGTH -- long edges stepping over land.
The right block thresholds grid nodes on DEGREE in one encoder/decoder subgraph --
grid nodes the model never reads from (g2m) or never writes back to (m2g).

Usage from the notebook:

    import figure2_builder as f2

    f2.use_png_renderer(scale=3)          # so 5.5 in is legible on screen
    fig = f2.build(BLOCKS, ZOOM=1.0)      # any config name is a keyword here
    f2.preview(fig)
    f2.export(fig, "figure2_ocean_defects")

Every module-level UPPERCASE name is a knob. `build(**overrides)` sets them for
the duration of one call and puts them back afterwards; `configure(**kw)` sets
them permanently. `show_config()` lists them (`config_keys()` returns the set), `show_config(common=True)` lists
just the ones worth touching. An unknown name raises rather than being ignored,
so a typo is not silently a no-op.

Encodings are the tool's, read from frontend/nextjs/src/lib/colors.json: gold
squares for grid nodes, mauve circles for mesh nodes sized by level, crimson G2M /
teal M2G / mauve M2M edges, and the blue-grey-red violation ramp. Never hardcode a
colour here -- colors.json is authoritative (see CLAUDE.md).
"""

from __future__ import annotations

import base64
import math
import os
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", str(ROOT / "backend" / "data"))
if str(ROOT / "backend" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend" / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch

from services.graph import load_graph, get_graph_metrics, grid_spacing_deg
from services.bsf_converter import is_bsf_format, convert_bsf_to_neural_lam

import json as _json
import re as _re

# ── the UI palette, read from the single source of truth ────────────────────
PAL = _json.loads((ROOT / "frontend/nextjs/src/lib/colors.json").read_text())

NODE_GRID    = PAL["mapNodes"]["grid"]        # gold  · grid nodes  (squares)
NODE_MESH    = PAL["mapNodes"]["mesh"]        # mauve · mesh nodes  (circles)
EDGE_ENCODER = PAL["mapEdges"]["encoder"]     # crimson · G2M
EDGE_DECODER = PAL["mapEdges"]["decoder"]     # teal    · M2G
EDGE_MESH    = PAL["mapEdges"]["mesh"]        # mauve   · M2M / inter-level
VIOL_ABOVE   = PAL["violation"]["above"]      # over threshold
VIOL_BELOW   = PAL["violation"]["below"]      # under threshold
VIOL_NEUTRAL = PAL["violation"]["neutral"]    # within range
TEXT         = PAL["ui"]["text"]
TEXT_MUTED   = PAL["ui"]["textMuted"]
TEXT_DIM     = PAL["ui"]["textDim"]
UI_BG        = PAL["ui"]["background"]
CHART_GREY   = "#9ca3af"                      # GNNExplorer.tsx: unthresholded chart colour
AXIS_LINE    = "#e2e8f0"                      # GNNExplorer.tsx: axis + tick stroke.
                                              #   The SCREEN stroke; on paper the charts use
                                              #   AXIS_COLOR -- see TYPOGRAPHY.

# The figure's type family, shared with figure3_builder so the two set as one plate.
# Not the UI's CSS stack: that starts with SF Pro, which macOS does not install as a
# normal font (only the hidden system ".SF NS"), so kaleido was falling through to
# Helvetica Neue here while matplotlib, given no family at all, fell to its own
# DejaVu Sans in figure3 -- and DejaVu's heavier stems made the same #64748b read as
# two different greys across the plate.
#
# Arial rather than Helvetica Neue, which is the closer match to the UI: macOS ships
# HelveticaNeue as a .ttc collection and matplotlib registers only its Regular face,
# so a bold request there silently returns the regular file. figure3's panel letters
# and its self-variable ranking ticks are bold, and the <b> on line ~2238 here is
# too. Arial ships Regular and Bold as separate files that BOTH renderers resolve,
# so bold is actually bold in both. figure3_builder.SANS is this same list.
SANS = "Arial, Liberation Sans, Helvetica, sans-serif"

MERC_MAX = 85.05113


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION -- every one of these is a valid keyword to build()
# ═══════════════════════════════════════════════════════════════════════════

GRAPH_SEARCH_DIRS = [
    ROOT / "notebooks" / "defect_graphs",
    ROOT / "data" / "defect_graphs" / "upload",
    ROOT / "backend" / "data" / "graphs",
    ROOT / "data" / "graphs",
    ROOT / "oceanography_graphs",
]
GRAPH_CACHE_DIR = ROOT / "notebooks" / ".graph_cache"

# ═══ GEOMETRY ═══════════════════════════════════════════════════════════════
# The figure is one row of four panels:
#
#   [ map A ] gap [ dist A ]   MIDGAP+rule   [ map B ] gap [ dist B ]
#
# Total width is fixed at FIG_W_IN. Every horizontal size is a FRACTION of it,
# so the ratios are what you tune and the inches take care of themselves. Height
# is set directly, because it is the one dimension the page does not constrain.

FIG_W_IN   = 5.50     # inches. Fixed by the column width; do not change.
MAP_H_IN   = 1.05     # inches. Panel height for BOTH maps -- with MAP_W this is
                      #   the map aspect, so the crop follows from it.

MAP_W      = 0.235    # map panel width, as a fraction of FIG_W_IN
DIST_W     = 0.170    # distribution panel width, same units
GAP_MD     = 0.046    # gap between a map and its own distribution. It also has to
                      #   clear the distribution's y tick labels, which render to the
                      #   LEFT of its domain -- too small and they land on the map.
GAP_MID    = 0.062    # gap between the two blocks -- deliberately wider, and the
                      #   rule below sits in the middle of it
FIT_WIDTH  = True     # rescale MAP_W / DIST_W proportionally so the row fills the
                      #   width exactly. False = leave the slack and print it.

# ── zoom ────────────────────────────────────────────────────────────────────
# A crop is given as a centre and a height in degrees of latitude. ZOOM divides
# that height, so >1 tightens the view and <1 pulls back; the longitude span
# follows automatically, so the panel aspect and the on-screen node size rule are
# both preserved. Set it globally here or per block with zoom= (they multiply).
# A block's shift=(dx, dy) then slides the crop off that centre -- +right / +up,
# in units of the crop's own width and height -- without changing its size.
ZOOM = 1.0

# Margins, in inches. Left holds the distribution y-axis furniture; bottom holds
# the x tick labels and axis title; top holds the panel titles.
MARGIN_IN  = dict(l=0.02, r=0.03, t=0.17, b=0.32)

# ── saving ──────────────────────────────────────────────────────────────────
# Two things happen to a figure on its way out, and they work together.
#
# SAVE_TRANSPARENT drops the white sheet the figure sits on. The panels, their
# ocean fills and the legend washes keep their own colours -- only the PAPER and
# the empty chart backgrounds go -- so the figure lands on whatever the page is
# instead of carrying a white rectangle onto it.
#
# SAVE_TIGHT then crops the file to what was actually drawn. The margins above are
# sized for the worst case (the tallest tick label, the longest axis title), so
# every figure otherwise carries a band of dead space that \includegraphics has to
# pay for. With the paper transparent the crop measures INK -- anything with any
# opacity at all -- rather than "not white", which is why the two belong together:
# on a white paper the trim can only ever reach the edge of the white.
#
# Neither re-renders or moves anything. The files are cropped after they are
# written: the raster is trimmed and the PDF's MediaBox narrowed, so the panels,
# type sizes and line weights are exactly the ones plotly drew.
SAVE_TRANSPARENT = True  # export with no background behind the figure
SAVE_TIGHT = True     # crop the saved file to the drawn content. False = save the
                      #   nominal canvas, exactly FIG_W_IN wide, slack included
SAVE_PAD   = 0.0      # inches kept around the content when SAVE_TIGHT. 0 cuts to
                      #   the ink; the figure's own breathing room belongs in the
                      #   page's \includegraphics, not baked into the file
SAVE_PROBE_SCALE = 2.0  # px per point of the throwaway render the crop is measured
                        #   on, when no PNG is being written to measure instead
SAVE_PDF_BLEED = 1.5  # points of slack left around the PDF crop only. The box is
                      #   measured on the RASTER, and kaleido rasterises and prints
                      #   through different text paths whose metrics disagree by
                      #   about a point -- measured, the print path runs wider. The
                      #   raster is cut exactly; the vector keeps this much so the
                      #   box can never shave a glyph. 0 to cut it exactly too

# ── the divider between the two blocks ──────────────────────────────────────
DIVIDER       = True
DIV_SCALE     = 1.00        # vertical extent as a fraction of the full panel span,
                            #   shrunk about that span's midpoint
DIV_DASH      = "1px,2.2px" # matches figure3's dotted rule
DIV_WIDTH     = 0.5
DIV_ALPHA     = 0.6

# ═══ THE TWO BLOCKS ═════════════════════════════════════════════════════════
# Content, not settings: blocks are passed to build(), which is why they are the
# one thing the notebook spells out in full. Block A thresholds on EDGE LENGTH --
# long mesh edges crossing land. Block B thresholds on NODE DEGREE -- grid nodes
# with no encoder (or decoder) edge. build.__doc__ repeats this key.
#
#   graph     folder name under GRAPH_SEARCH_DIRS, or an absolute path
#   subgraph  registry key: "m2m" | "m2m_0".. | "g2m" | "m2g" | "mesh_up_0"..
#   metric    "length"   -> threshold edges, chart = edge-length histogram
#             "grid_deg" -> threshold GRID nodes on their degree in this subgraph.
#                           Resolves to the source side for g2m (out-degree) and
#                           the target side for m2g (in-degree), so the block does
#                           not have to know which way the subgraph points.
#             "mesh_deg" -> same, for the mesh side (m2m_*, mesh_up/down_*)
#             "src_deg" / "tgt_deg" -> the explicit sides, still accepted
#   zoom      crop magnification for this block; multiplies the global ZOOM
#   lo, hi    threshold values in the metric's own units (km, or integer degree).
#             None on a side = that side of the padded full range, i.e. inactive.
#   hi_control  a SECOND cut on the same metric, beside hi -- the control case. The
#             band between the two (what one cut flags and the other does not) draws
#             in a lighter shade of the flag colour, in the chart and on the map, and
#             the key gains a row naming it. Requires hi; may sit either side of it.
#   edge_alpha  optional 0-255 override for this block's edges. None = the default
#             for its metric (EDGE_ALPHA_NORMAL for length, DEGREE_BLOCK_EDGE_ALPHA
#             for a degree, SUBGRAPH_EDGE_ALPHA in full view). 0 hides them.
#   settings  dict of SETTING -> value applied to this block alone (marks, charts,
#             key, crop). Layout settings are figure-wide and rejected here.
#   legend_loc  corner for this panel's key: "bottom-left" | "bottom-right" |
#             "top-left" | "top-right" | "none". Default LEGEND_LOC.
#   show_subgraph  True draws the whole subgraph the degree was counted over, not
#             just the nodes: semantic edge colour, endpoints at full size, and
#             arrowheads if the crop is small enough to earn them. Default
#             SUBGRAPH_VIEW. Ignored on a length block.
#   region    key into REGIONS below, or None to use `crop`
#   crop      explicit dict(center=(lat, lon), height=<deg lat>) -- overrides region
#   shift     (dx, dy) to slide the crop off that centre without resizing it:
#             +dx right, -dx left, +dy up, -dy down, in units of the crop's own
#             width and height. (0.5, 0) is half a panel to the right. Applied
#             after zoom=, so a nudge means the same thing at any magnification.
#   title     panel title above the map; None -> auto from graph + subgraph

# Threshold values are chosen against the metric's real distribution, which
# metric_report() prints. On m2m_2, 68 of 5,770 undirected edges are >= 700 km
# and the longest is 1,333 km; on g2m, lo=0 isolates the 1,183 grid nodes with no
# encoder edge at all (lo=1 would sweep in 257,667 legitimate degree-1 nodes).
# ── the encoder/decoder side the grid-node block interrogates ───────────────
# g2m asks which grid nodes never reach the mesh (out-degree 0, the encoder blind
# spot); m2g asks which are never written back to (in-degree 0, the decoder blind
# spot). Different defects, and in general different nodes: on prod_cluster_noconnect
# 1,183 grid nodes have no g2m edge and 179 no m2g edge, only 128 have neither.
# metric="grid_deg" picks the right side of whichever subgraph a block names, so
# switching sides is one edit to `subgraph=`.
#
# This is a lookup for the caller to title its block with -- NOT a setting. The
# side lives in the block spec, because that is what the figure is actually built
# from; a module-level switch would be read once at import and then be a lie.
GRID_BLOCK_TITLE = dict(g2m="Grid nodes with no encoder edge",
                        m2g="Grid nodes with no decoder edge")

# The default content, used when build() is called with no blocks. The notebook
# passes its own; this is what makes `import figure2_builder as f2; f2.build()` work.
DEFAULT_BLOCKS = [
    dict(graph="prod_cluster_default", subgraph="m2m_2", metric="length",
         lo=None, hi=700.0, region="white_sea", zoom=1.0,
         title="Mesh edges crossing land"),
    # zoom=2 is not cosmetic: at the catalogue height this crop holds ~10k g2m edges,
    # which fuse into bands. Halving the span separates them into the texture the
    # flagged nodes need to read against.
    dict(graph="prod_cluster_noconnect", subgraph="g2m", metric="grid_deg",
         lo=0, hi=None, region="canadian_arctic", zoom=2.0,
         title=GRID_BLOCK_TITLE["g2m"]),
]

# ═══ CROP PRESETS ═══════════════════════════════════════════════════════════
# Centres found by scripts/audit_graph_defects.py --regions. `height` is degrees
# of latitude spanned top to bottom; the longitude span follows from the panel
# aspect, so a cell stays square exactly as it does in the tool.
# See the REGIONS table cell below for what each one contains.

REGIONS = {}          # filled in by the regions cell, two cells down

# ═══ CHART STYLE ════════════════════════════════════════════════════════════
LEN_BINS        = 22       # edge-length histogram bins
LEN_CLUSTER_GAP = 30.0     # km -- UI clusterByGap() for discrete-length subgraphs
DEG_CHART_STYLE = "lollipop"   # "lollipop" (UI encoding) | "bar"
# ── lollipop heads ─────────────────────────────────────────────────────────
# The head is sized from the horizontal pitch between adjacent items rather than
# fixed. A mesh_0 degree chart carrying ten degrees packs them at half the pitch of
# a five-degree grid chart in the same panel width, and a head that reads correctly
# in the second one merges into a caterpillar in the first. LOLLI_HEAD_PX is the
# size at generous spacing AND the cap -- the head only ever shrinks from it, so a
# sparse chart looks exactly as it did before.
LOLLI_HEAD_PX   = 5.0      # head diameter in points at generous spacing; also the cap
LOLLI_HEAD_FRAC = 0.42     # ... and as a fraction of the pitch between adjacent items,
                           #   whichever of the two is smaller
LOLLI_HEAD_MIN  = 2.0      # floor -- below this the ring reads as a thick stem end
LOLLI_STEM_W    = 1.5      # stem width in points. The stem is the count; the head only
                           #   marks where it ends, so a stem heavier than the head's
                           #   ring turns the chart into a bar chart with knobs on.
LOLLI_HEAD_W    = 1.2      # width of the head's ring, in points. The head is hollow --
                           #   this is the only ink in it -- so it also sets how solid a
                           #   small head reads: at LOLLI_HEAD_MIN a ring much above 1.2
                           #   fills in and the dot stops being a ring at all.
LOLLI_X_PAD     = 1.0      # left pad on a lollipop axis, in HEAD DIAMETERS. The first
                           #   item usually sits at degree 0, which is the axis's own
                           #   start, so without this its head is cut in half by the y
                           #   axis. 0 restores the old flush start.
DEG_THR_OFFSET  = 0.5      # where a degree threshold's dashed marker sits, in degrees.
                           #   Degrees are integers, so a marker drawn exactly on lo
                           #   cuts through the lollipop it describes. 0.5 puts it in
                           #   the empty space between lo and lo+1, which is the real
                           #   boundary between flagged and neutral; the hi marker is
                           #   mirrored to hi - offset. 0 draws on the value itself.
THR_LINE_W      = 0.9      # width of a threshold marker in the distribution, in px.
                           #   A marker is an annotation on the chart, not a rule
                           #   across it; and a panel carrying a control threshold
                           #   draws two of them side by side.

# ── a control threshold ────────────────────────────────────────────────────
# hi_control= on a block sets a SECOND cut on the same metric, beside hi. It changes
# nothing about the claim: hi is still the block's threshold and still draws in the
# full flag colour. What the control adds is the BAND BETWEEN THE TWO -- the marks one
# cut flags and the other does not -- drawn in a lighter shade of that same colour, in
# the chart and on the map alike. A 20 km shift in where the line is put then shows as
# marks changing colour, instead of being asserted in a caption.
CONTROL_TINT    = 0.42     # how much of the flag colour survives the mix toward white.
                           #   1 is indistinguishable from a real flag, 0 erases it.
CONTROL_SUFFIX  = ""       # appended to the control row's generated legend label. Empty
                           #   by default: the two rows already differ in the number they
                           #   name and in the shade they name it with, and a panel whose
                           #   whole point is "which of these two cuts" gains nothing from
                           #   a word saying which is which. Set it to " (control)" to
                           #   spell it out; an explicit legend_labels["hi_control"]
                           #   replaces the row's text outright.

CHART_LOG_Y     = True     # degree and length counts are heavy-tailed; log y keeps
                           #   the flagged tail visible next to the bulk
DIST_TITLE      = True     # small title over each distribution panel
# ── the distribution column, split ─────────────────────────────────────────
# A dual-claim block thresholds two things, so one chart beside it can only show
# half of what the panel asserts. DIST_SPLIT stacks a chart per claim in the same
# column: the edge metric on top, the node degree underneath, in the order the
# caption reports them.
#
# The gap carries the top chart's tick labels AND its axis title AND the lower
# chart's own title, which is three rows of type: measured, 0.22 collides outright
# and 0.30 still grazes, so 0.38 is the default. Below ~0.34 expect overlap. The
# charts pay for it -- each half is (1 - gap)/2 of the row, about 23 px at 0.38 --
# which is the real cost of asking one column to carry two claims.
DIST_SPLIT      = True     # split the column when a block sets node_metric
DIST_SPLIT_GAP  = 0.38     # gap between the halves, as a fraction of the row height
SHOW_Y_TITLE    = False    # "count" on the y axis. Off by default: at 5 pt it costs
                           #   more width than it explains.
LEN_CHART       = "hist"   # "auto" follows the registry (lollipop for graphs whose
                           #   edge lengths are discrete, histogram otherwise). A k-means
                           #   cluster mesh has continuous lengths but is still tagged
                           #   lollipop, so force "hist" for the ocean graphs.
FLAG_COUNT_SCOPE = "crop"  # "crop" counts what the panel shows; "graph" counts globally

# ═══ TYPOGRAPHY ═════════════════════════════════════════════════════════════
# Nominal px == points (PX_PER_IN below is 72), so these are point sizes and the
# PDF comes out exactly FIG_W_IN wide.
#
# Two base sizes and four ROLES on top of them -- the same names, the same values
# and the same meanings as figure3_builder, so the two figures set as one plate and
# neither has to be eyeballed against the other. A role fixes a size AND a colour;
# everything drawn here belongs to exactly one of them:
#
#   TITLE    every panel title -- a map's and a distribution's alike, in one
#            size and one colour, so no panel reads as subordinate to another
#   LEGEND   the in-map key AND the "n of m flagged" line under the map: both
#            are captions on the map, so both are set as captions
#   TICKNUM  the numbers on a chart's axes, x and y
#   AXLAB    the axis title under those numbers (and "count", when SHOW_Y_TITLE)
#
# configure(FONT_S=..) / configure(FONT_XS=..) fan out to every role that defaults
# to them, so the whole figure rescales from one keyword. Naming a role in the same
# call wins over the fan-out: configure(FONT_XS=5, TITLE_FS=8) does what it says.
FONT_S, FONT_XS = 7.0, 6.0     # figure3_builder's two base sizes, verbatim

TITLE_FS   = FONT_S            # panel titles, maps and distributions
LEGEND_FS  = FONT_XS           # in-map key, and the flagged-count line under it
TICKNUM_FS = FONT_XS           # numeric ticks, both axes
AXLAB_FS   = FONT_XS           # axis titles under numeric ticks

# Colours are fixed per role; only the sizes above vary.
TITLE_FC   = TEXT
LEGEND_FC  = TEXT_MUTED
TICKNUM_FC = TEXT_MUTED
AXLAB_FC   = TEXT_MUTED

# ── chart axis chrome ───────────────────────────────────────────────────────
# The distributions' own x and y axes. figure3_builder draws its bar-chart spines
# in a soft black rather than the UI's slate stroke, and the same holds here: at
# 0.5 pt on white paper #e2e8f0 is barely a line at all, so the chart loses its
# frame of reference while the screen version keeps one.
AXIS_COLOR    = "rgba(0,0,0,0.333)"   # figure3_builder's "#00000055"
AXIS_WIDTH    = 0.5                   # spine and tick stroke, in points
AXIS_TICK_LEN = 1.6                   # tick length, in points
AXIS_ARROWS   = True                  # filled arrowhead at the far tip of each axis --
                                      #   x to the right, y at the top, as in figure3
AXIS_ARROW_PX = 2.6                   # its length in points, apex on the axis end

# ── panel letters ───────────────────────────────────────────────────────────
# (a) (b) (c) ... for a paper that refers to panels in prose. Ordered the way a
# reader scans: left to right across the row, and top to bottom inside a split
# distribution column, so a dual-claim block's two charts are consecutive.
# The eleven names below are the SAME in figure2_builder and figure3_builder --
# same values, same meanings -- so one plate can be lettered from one set of
# settings. Only PANEL_LETTER_LOC's default differs, and only because each
# figure's own title geometry forces it; see there.
PANEL_LETTERS      = False      # off by default; the figure stands alone without them
PANEL_LETTER_SCOPE = "all"      # "all" letters every panel | "maps" only the maps. A
                                #   skipped panel does not consume a letter either, so
                                #   the sequence never has gaps.
PANEL_LETTER_START = "a"        # first letter, so a multi-figure plate can continue
PANEL_LETTER_FMT   = "({})"     # "({})" -> (a) | "{}" -> a | "{}." -> a.
PANEL_LETTER_LOC   = "inside"   # "inside" = the panel's own top corner, over the map or
                                #   the chart. "above" = the title's baseline, outside the
                                #   panel. THIS figure defaults to "inside": its titles are
                                #   centred and run the full panel width, so "above"
                                #   renders as "(a)Mesh edges crossing land". Measured, not
                                #   assumed. figure3_builder defaults to "above", where the
                                #   titles are short enough to leave that line free.
PANEL_LETTER_SIDE  = "left"     # "left" | "right" -- which corner. Honoured by both LOCs;
                                #   "right" anchors the letter's RIGHT edge to the panel's
                                #   right edge, so it never overhangs.
PANEL_LETTER_PAD   = (0.0, 0.0) # (dx, dy) in points: +x right, -x left, +y up, -y down.
                                #   The nudge on top of whatever LOC and SIDE resolve to
                                #   -- use it to clear a legend or a tall first bar without
                                #   moving to the other corner. Replaces the old
                                #   PANEL_LETTER_INSET_PX, which only moved "inside"
                                #   letters and only diagonally.
                                #   ONE pair pads every letter the same. A LIST of pairs
                                #   pads them individually, in letter order a, b, c ...,
                                #   and must hold exactly one entry per letter -- e.g.
                                #   [(0,0), (0,0), (0,-9), (0,0), (-11,0)] to drop (c)
                                #   under a top-left key and pull (e) left of a legend.
                                #   A list of the wrong length raises.
PANEL_LETTER_SIZE  = None       # point size; None = TITLE_FS, the panel title's size
PANEL_LETTER_BOLD  = True
PANEL_LETTER_WASH  = True       # white pad behind an "inside" letter. The top corner of a
                                #   histogram is exactly where its tallest bars are, so
                                #   without this the letter reads on top of the data.
PANEL_LETTER_HEADROOM = 1.16    # extra y range on a lettered CHART, "inside" only, so the
                                #   letter sits in reserved space rather than on the marks.
                                #   Without it the wash clips the tallest bar or the first
                                #   lollipop, which is occluding data to make room for a
                                #   label. Maps need none -- a letter over the basemap
                                #   hides nothing.

SHOW_LEGENDS = True        # the small key inside each map
LEGEND_SCALE = 1.0
LEGEND_SUBGRAPH = True     # on a degree panel, name the subgraph the degree was
                           #   counted over and show it in the colour it draws in
LEGEND_EDGE_ALPHA = 200    # the key's edge swatch is one 1 px line, not thousands, so it
                           #   is drawn near-opaque even though the map's edges are faint
SLIM_LEGEND  = False       # the key names ONLY what the block thresholded, and names it
                           #   in the metric's own terms: "grid point degree = 0",
                           #   "mesh edge >= 200 km" -- one row per claim, and for a
                           #   dual-claim block one row per claim. The "within range"
                           #   row and the LEGEND_SUBGRAPH context row are dropped: on a
                           #   panel whose whole assertion is the flagged marks, they
                           #   spend two of four rows on what is NOT being claimed.
                           #   An explicit legend_labels entry still wins, so a block can
                           #   override any single row it phrases better itself.
LEGEND_LOC   = "bottom-left"   # default corner for the in-map key. Per block with
                               #   legend_loc=, because which corner is empty is a
                               #   property of the crop, not of the figure:
                               #   "bottom-left" | "bottom-right" | "top-left" |
                               #   "top-right" | "none" (suppress on that panel).
LEGEND_INSET_PX = 2.0          # gap between the key and the panel frame, in points

# ── the key's card ──────────────────────────────────────────────────────────
# It used to be a plain white wash with no edge, on the argument that at 5 pt a
# border costs more attention than it returns. At the role sizes above the key is
# large enough to want a boundary: without one it dissolves into whatever the crop
# happens to put behind it. So it is a card -- rounded, and outlined in a grey
# light enough to read as an edge rather than as a mark. figure3_builder's legend
# takes the same colour, from LEGEND_BORDER there.
LEGEND_BG        = "rgba(255,255,255,0.86)"
LEGEND_BORDER    = "#e8ebee"   # really light grey: the edge is there to bound the
                               #   card, not to be seen doing it
LEGEND_BORDER_W  = 0.5         # points
LEGEND_RADIUS_PX = 2.5         # corner radius, in points; clamped to half the box

# ═══ MARK GEOMETRY (from GNNExplorer.tsx / LeftPanel.tsx) ═══════════════════
NODE_SIZE_M           = 60_000
REFERENCE_SPACING_DEG = 1.5
NODE_SIZE_SCALE       = 1.0
NODE_SIZE_MIN_PX      = 0.30   # the 0.25 deg ocean grid is denser than the page: at
NODE_SIZE_MAX_PX      = 3.2    #   these sizes neutral nodes read as texture, not marks
MESH_NODE_MIN_PX      = 1.7    # floor for a COUNTERPART endpoint. base_node_px() sizes marks
                               #   from the GRID spacing, and mesh_0 sits ~1.4 deg apart
                               #   against a 0.25 deg grid -- six times sparser -- so the
                               #   grid's rule drives a mesh node to 0.27 px, i.e. invisible.
                               #   Without this the endpoints of g2m/m2g are simply not on
                               #   the page, and the panel reads as grid nodes alone.
MESH_SIZE_SCALE       = 2 / math.sqrt(math.pi)
MESH_NODE_COARSE_BOOST, MESH_NODE_FINE_BOOST = 3.5, 0.8
MESH_STROKE_COARSE,     MESH_STROKE_FINE     = 3.5, 0.8
NODE_BORDER_FRAC      = 1 / 24
NODE_BORDER_MIN_PX    = 0.25
NODE_ICON_ALPHA         = 220
NODE_VIOL_ALPHA         = 235
VIOL_NODE_BOOST         = 2.6  # flagged nodes draw larger than neutral ones. Without this
                               #   a defect is invisible at the density above.
NODE_VIOL_NEUTRAL_ALPHA = 60   # the unflagged field. Low on purpose: at 0.25 deg the
                               #   grid fills the panel, and anything heavier reads as
                               #   banding rather than as background
EDGE_WIDTH_SCALE        = 0.42
EDGE_ALPHA_NORMAL       = 80
DEGREE_BLOCK_EDGE_GREY  = True # Draw a degree block's context edges through the tool's own
                               #   grey_of() de-emphasis instead of their semantic colour.
                               #   colors.json gives the encoder and the "over threshold"
                               #   violation the SAME crimson, and the decoder and "under
                               #   threshold" the SAME cyan -- so on an m2g degree panel the
                               #   flagged nodes and the edges they are missing come out the
                               #   same colour and the claim disappears. Greying the edges is
                               #   the tool's existing answer to "present but not the subject".
                               #   False restores the semantic colour.
DEGREE_BLOCK_EDGE_ALPHA = 60   # a degree block draws the subgraph the degree was counted
                               #   over -- without it the panel shows isolated nodes with no
                               #   visible thing to be isolated FROM. Greyed edges tolerate a
                               #   higher alpha than coloured ones did; on the 0.25 deg grid
                               #   they still read as texture, so pair a dense crop with zoom=.
                               #   Override per block with edge_alpha=; 0 drops them.
EDGE_ALPHA_VIOL_NEUTRAL = 40
EDGE_ALPHA_VIOL_FLAG    = 235

# ── showing a degree block's subgraph in full ───────────────────────────────
# A degree panel's default is de-emphasis: greyed edges, mesh endpoints present but
# subordinate, so the flagged nodes carry the panel. SUBGRAPH_VIEW flips it to the
# tool's Explore encoding of that same subgraph -- semantic edge colour, endpoints
# at full size, arrowheads on a directed subgraph -- so the reader can see WHAT the
# isolated nodes are isolated from. Per block with show_subgraph=True.
#
# It overrides DEGREE_BLOCK_EDGE_GREY, and on m2g that reintroduces the collision
# that flag exists to avoid: colors.json gives the decoder and violation-below the
# same cyan, so flagged nodes and the edges they lack come out one colour. Fine when
# the panel's claim IS the subgraph; keep it False when the claim is the defect.
SUBGRAPH_VIEW        = False
SUBGRAPH_EDGE_ALPHA  = 70    # edges in full view. Higher than DEGREE_BLOCK_EDGE_ALPHA
                             #   because they are the subject now, not the background.
SUBGRAPH_ARROWS      = True  # arrowheads at the target end of g2m / m2g, so the
                             #   direction that makes deg_out and deg_in different
                             #   is visible rather than asserted in the title
SUBGRAPH_ARROW_FRAC  = 0.85  # arrowhead size as a fraction of the mesh node size
SUBGRAPH_ARROW_MAX_E = 600   # give up on arrowheads past this many edges in the crop.
                             #   One head per edge is a solid mass at 4,000 edges and
                             #   says less than no heads at all; zoom in to earn them.
                             #   region_table() and the flagged-count line both report
                             #   crop contents, so this is easy to aim for.

# ═══ BASEMAP ════════════════════════════════════════════════════════════════
BASEMAP_ALPHA = 0.55
GEO = dict(
    resolution=50, landcolor="#eceff2", oceancolor="#dde5ea", lakecolor="#dde5ea",
    coastlinecolor="#b7c2cc", coastlinewidth=0.45,
    countrycolor="#dfe5ea", countrywidth=0.3, showcountries=False,
    framecolor="#cbd5e1", framewidth=0.7,
)

# ── a finer basemap than plotly can draw ───────────────────────────────────
# layout.geo.resolution accepts only [110, 50] -- 1:110m and 1:50m Natural Earth --
# and 50 is already the finer of the two. There is no built-in 10m.
#
# MEASURED, on the canadian_arctic crop at zoom 3: plotly at 50 and Natural Earth at
# 10m are visually equivalent (66,130 vs 66,450 coastline edge pixels, 0.5% apart).
# 50 is enough. Keep BASEMAP_SOURCE="plotly" unless you have a specific reason.
#
# The one thing this path buys is INDEPENDENCE from plotly.js's topojson loading.
# Kaleido fetches the 50m topojson on demand and silently falls back to 110m when it
# cannot -- which produced visibly coarser exports here until the file was cached,
# with no error and no warning. If your exports ever come out blocky, that is why,
# and this is the way around it.
#
# BASEMAP_SOURCE="naturalearth" reads the 1:10m shapefiles through cartopy instead
# and draws land and lakes as clipped, filled Scattergeo polygons underneath
# everything else, with plotly's own land/lakes/coastlines switched off so the two
# cannot disagree. The ocean fill and the frame stay plotly's.
#
# Costs, so this is opt-in: it needs cartopy + shapely, the shapefiles download once
# (~7 MB, cached by cartopy), and it multiplies basemap vertices about 4x for a small
# Arctic crop -- 1,025 -> 4,373 -- which lands in the PDF. Worth it on a tight crop,
# wasted on a whole-world one.
BASEMAP_SOURCE = "plotly"      # "plotly" | "naturalearth"
BASEMAP_NE_RES = "10m"         # "10m" | "50m" | "110m" -- only read when the
                               #   source is "naturalearth"

PX_PER_IN = 72        # nominal px == PDF points. Leave alone.

# ═══ COMPATIBILITY ══════════════════════════════════════════════════════════
# The loading and mark-drawing cells are copied verbatim from figure2_builder so
# they stay in step with it. They also serve that notebook's Explore panel, which
# this figure does not have, so the handful of names it needs are defined here.
# EXPLORE_SUBGRAPHS = [] makes build_row() skip cropping every subgraph, which is
# the expensive half of it.
EXPLORE_SUBGRAPHS = []
EXPLORE_ARROWS    = False
ARROW_SIZE_FRAC   = 0.7
NODE_DEGREE_SIDE  = "src"        # only used by figure2_builder's degree_panel()
NODE_CHART_STYLE  = DEG_CHART_STYLE
CROP = dict(center=(0.0, 0.0), height=60.0, aspect=1.0)   # fallback; every block sets its own


# ═══════════════════════════════════════════════════════════════════════════
# CROP PRESETS, BY FAILURE MODE
# ═══════════════════════════════════════════════════════════════════════════
# Three failure modes, three classes of crop. A region is filed under the mode it
# was FOUND for; the table in the next cell measures all three in every crop, so
# a region that happens to carry two shows up there as carrying two.
#
#   length   long mesh edges stepping over land   metric="length"   on m2m_*
#   g2m_iso  grid nodes with no ENCODER edge      metric="grid_deg" on g2m
#   m2g_iso  grid nodes with no DECODER edge      metric="grid_deg" on m2g
#
# Centres came from
#     python scripts/audit_graph_defects.py <graph> --regions --cell 12
# on the production-resolution graphs in data/defect_graphs/upload/, and the
# isolation counts in the notes were then measured in the crop itself, at the
# height given here and the figure's own panel aspect. `height` is degrees of
# latitude; the longitude span follows from the aspect. Point a block at one with
# region="<key>", or override with crop=dict(center=(lat, lon), height=<deg>).

CLASSES = {
    "length":  "long mesh edges over land   ·  metric='length'   on m2m_*",
    "g2m_iso": "no encoder edge (deg_out 0) ·  metric='grid_deg' on g2m",
    "m2g_iso": "no decoder edge (deg_in 0)  ·  metric='grid_deg' on m2g",
    "other":   "mesh isolation and whole-graph fragmentation",
}

REGIONS = {
    # ═══ 1 · land-crossing mesh edges ═══════════════════════════════════════
    # On prod_cluster_default (identical to _highcut -- see the note in §3).
    # Counts below are the audit script's land test, not a length threshold: an
    # edge crosses land when a sample of its arc does, which is why a crop can
    # hold land-crossing edges that no single `hi` isolates.
    "indonesia":       dict(cls="length", center=(-5.1, 125.0), height=22.0,
                            note="densest land-crossing cluster: 12 edges at m2m_0, 7 at m2m_1, 4 at m2m_2"),
    "java_flores":     dict(cls="length", center=(-6.1, 115.7), height=16.0,
                            note="11 edges at m2m_0 -- Java Sea / Flores Sea straits"),
    "solomon":         dict(cls="length", center=(-6.4, 152.9), height=16.0,
                            note="8 edges at m2m_0; worst single edge 136 km / 61% over land"),
    "philippines":     dict(cls="length", center=(6.7, 124.3), height=18.0,
                            note="6 edges at m2m_0, 4 at m2m_1 -- Sulu / Celebes Sea"),
    "svalbard":        dict(cls="length", center=(78.4, 14.7), height=9.0,
                            note="the worst m2m_0 edge: 250 km, 70% over land, across Spitsbergen"),
    "alaska_pen":      dict(cls="length", center=(56.0, -160.0), height=14.0,
                            note="5 edges at m2m_0; a 172 km edge 61% over the Alaska Peninsula"),
    "japan":           dict(cls="length", center=(34.3, 133.1), height=13.0,
                            note="m2m_1: a 327 km edge 70% over land, across Honshu"),
    "baffin":          dict(cls="length", center=(63.8, -68.3), height=16.0,
                            note="m2m_1: a 439 km edge 61% over land, across Baffin Island"),
    "scotland":        dict(cls="length", center=(58.0, -3.8), height=10.0,
                            note="m2m_1: a 351 km edge 52% over land, across Scotland"),
    "bahamas":         dict(cls="length", center=(25.4, -77.7), height=12.0,
                            note="m2m_2: a 445 km edge 52% over land, across the Bahamas bank"),
    "white_sea":       dict(cls="length", center=(67.7, 44.6), height=17.0,
                            note="m2m_2: an 825 km edge 43% over land, Kara Sea to the White Sea"),
    "north_sea_baltic": dict(cls="length", center=(55.5, 6.0), height=16.0,
                            note="Ireland to the Gulf of Riga. The dual-threshold crop: on "
                                 "m2m_0, 8 undirected edges >=200 km hop Denmark, Wales and "
                                 "the Channel, AND 7 mesh nodes have degree <=1 (3 of them 0) "
                                 "-- five in the Danish straits, which is the Baltic losing "
                                 "its connection to the North Sea"),
    "panama":          dict(cls="length", center=(9.5, -80.0), height=13.0,
                            note="the isthmus: edges hop Central America into the Caribbean. "
                                 "In crop, 16 m2m_0 >=150 km, 7 m2m_1 >=300 km, 5 m2m_2 >=600 km "
                                 "-- the level's own scale matters here, the 700 km cutoff the "
                                 "figure uses on m2m_2 flags none of them (the longest is 689 km)"),
    "ecuador":         dict(cls="length", center=(-1.7, -80.6), height=14.0,
                            note="shipped global_cluster_analysis: 286 km, 70% over land across Ecuador"),
    "caribbean":       dict(cls="length", center=(21.4, -76.3), height=14.0,
                            note="m2m_1: 5 edges over Cuba / Hispaniola"),
    "north_sea":       dict(cls="length", center=(53.4, -3.4), height=12.0,
                            note="m2m_1: 5 edges across England and the Danish straits"),

    # ═══ 2 · grid nodes with no ENCODER edge (g2m, deg_out == 0) ═════════════
    # prod_cluster_noconnect: 1,183 of 675,219 grid nodes graph-wide. Counts are
    # g2m-isolated / m2g-isolated inside this crop.
    "canadian_arctic": dict(cls="g2m_iso", center=(79.7, -88.1), height=9.0,
                            note="104 / 43 -- Sverdrup Basin; the densest g2m cluster there is"),
    "baffin_channels": dict(cls="g2m_iso", center=(67.2, -78.4), height=13.0,
                            note="122 / 21 -- Foxe Basin, Fury and Hecla Strait"),
    "kara_ob":         dict(cls="g2m_iso", center=(72.6, 77.2), height=13.0,
                            note="121 / 16 -- Kara Sea shelf, Gulf of Ob and Yenisei Gulf. A "
                                 "superset of ob_yenisei: same 117 flagged nodes plus 7 further "
                                 "out on the shelf. Use ob_yenisei to fill the frame with them"),
    "greenland_sea":   dict(cls="g2m_iso", center=(77.4, -20.9), height=13.0,
                            note="94 / 19 -- east Greenland fjords and Fram Strait"),
    "east_siberian":   dict(cls="g2m_iso", center=(74.3, 112.8), height=13.0,
                            note="61 / 3 -- Laptev and East Siberian shelf; almost purely g2m"),
    "ob_yenisei":      dict(cls="g2m_iso", center=(70.5, 77.0), height=8.0,
                            note="116 / 14 -- the southern Kara Sea, both estuary arms in one "
                                 "frame: the Gulf of Ob (66 flagged) and the Yenisei Gulf (28), "
                                 "plus the Taz Estuary (6). The tightest framing of this cluster; "
                                 "kara_ob is the same place wider, and holds 7 more"),

    # ═══ 3 · grid nodes with no DECODER edge (m2g, deg_in == 0) ══════════════
    # prod_cluster_noconnect: 179 of 675,219 graph-wide -- an order of magnitude
    # rarer than the encoder defect, so these crops are tighter. Same counts.
    # ob_estuary used to sit here; it framed exactly the same 117 nodes as
    # ob_yenisei, and 116 of those are g2m, so it was never an m2g crop.
    "nares_strait":    dict(cls="m2g_iso", center=(80.6, -85.0), height=7.0,
                            note="93 / 42 -- Nares Strait and Kane Basin; the densest m2g cluster"),
    "salish":          dict(cls="m2g_iso", center=(49.6, -124.5), height=7.0,
                            note="11 / 12 -- Strait of Georgia and Puget Sound; the one crop where "
                                 "m2g outnumbers g2m"),
    "magellan":        dict(cls="m2g_iso", center=(-53.4, -70.3), height=9.0,
                            note="34 / 12 -- Strait of Magellan and Tierra del Fuego"),
    "alexander_arch":  dict(cls="m2g_iso", center=(57.6, -134.2), height=9.0,
                            note="22 / 11 -- Alexander Archipelago, the SE Alaska inside passage"),

    # ═══ other · mesh isolation and fragmentation ════════════════════════════
    "baltic_mesh":     dict(cls="other", center=(55.6, 11.1), height=9.0,
                            note="prod_cluster_default: 2 isolated MESH nodes, Danish straits "
                                 "(metric='mesh_deg' on m2m_0)"),
    "amundsen":        dict(cls="other", center=(69.6, -114.1), height=12.0,
                            note="prod_cluster_default: 2 isolated mesh nodes, Amundsen Gulf"),
    "chile_fjords":    dict(cls="other", center=(-42.4, -73.1), height=12.0,
                            note="prod_cluster_default: 1 isolated mesh node, Chilean fjords"),
    "north_atlantic":  dict(cls="other", center=(45.0, -35.0), height=60.0,
                            note="prod_cluster_lowcut: open-ocean fragmentation, no coast to explain it"),
    "global":          dict(cls="other", center=(10.0, 0.0), height=150.0,
                            note="whole world; use with prod_cluster_lowcut, which is shredded everywhere"),
}

def region(key: str) -> dict:
    if key not in REGIONS:
        raise KeyError(f"unknown region '{key}'. Available: {sorted(REGIONS)}")
    r = REGIONS[key]
    return dict(center=r["center"], height=r["height"], aspect=1.0)   # aspect set later

def regions_of(cls: str) -> list[str]:
    return [k for k, v in REGIONS.items() if v["cls"] == cls]


# ═══════════════════════════════════════════════════════════════════════════
# REGION x FAILURE MODE -- config for region_table()
# ═══════════════════════════════════════════════════════════════════════════
# Which of the three defects each crop actually contains. The catalogue's classes
# are curation -- where each crop came from; this is measurement, and the two do
# not always agree (baffin was found for a land-crossing edge and turns out to be
# one of the densest g2m clusters as well).
#
# Edges are counted undirected, nodes as nodes, both restricted to the crop, using
# the same OR-semantics window the panels use: an edge is in when EITHER endpoint
# is in, which is what keeps a long edge that leaves the frame from vanishing.

# One graph per mode: the length defect lives in the as-built graph, the isolation
# defects only exist in the sabotaged one, so a single graph cannot show all three.
MODE_GRAPHS = dict(length="prod_cluster_default",
                   g2m_iso="prod_cluster_noconnect",
                   m2g_iso="prod_cluster_noconnect")

# What counts as "long" at each mesh level. A single number would be meaningless
# across levels -- m2m_0 tops out at 563 km and m2m_2 reaches 1,333 -- so each level
# gets its own, at roughly the top 0.1-0.3% of that level's own lengths. m2m_2's is
# the value BLOCKS thresholds at, so the table and the figure agree.
LONG_EDGE_KM = {"m2m_0": 250.0, "m2m_1": 400.0, "m2m_2": 700.0}

# How much of a defect makes a crop worth pointing a panel at. Below these the
# letter is withheld: one stray isolated node is a curiosity, not a figure.
MODE_MIN_EDGES = 1
MODE_MIN_NODES = 5


# ═══════════════════════════════════════════════════════════════════════════
# THE KNOBS -- which names build() and configure() accept
# ═══════════════════════════════════════════════════════════════════════════
# Anything UPPERCASE defined above, minus the handful that are not settings:
# ROOT and PAL are facts about the checkout, MERC_MAX is a property of the
# projection, and DEFAULT_BLOCKS is the fallback content, passed positionally.

_NOT_CONFIG = {"ROOT", "PAL", "MERC_MAX", "SANS", "DEFAULT_BLOCKS",
               "GRID_BLOCK_TITLE", "CLASSES", "REGIONS"}

def config_keys() -> frozenset[str]:
    """
    Every name build() and configure() accept, computed from the module's CURRENT
    globals on each call.

    Deliberately not a frozen constant. Under %autoreload 2 -- which the notebook
    uses -- function bodies are patched in place but module-level data is not
    re-evaluated, so a constant computed at first import goes stale the moment a
    setting is added, and the next build() rejects a name that plainly exists in
    the file. Computing it on demand costs nothing and cannot drift.
    """
    return frozenset(
        k for k, v in list(globals().items())
        if k.isupper() and not k.startswith("_") and k not in _NOT_CONFIG
        and not callable(v) and not isinstance(v, type(sys))
    )

# The knobs worth reaching for. Everything else is a detail of the tool's
# encodings that you change only to match a UI change.
COMMON_KEYS = (
    "ZOOM",
    "FIG_W_IN", "MAP_H_IN", "MAP_W", "DIST_W", "GAP_MD", "GAP_MID", "MARGIN_IN",
    "LEN_BINS", "CHART_LOG_Y", "DIST_TITLE", "SHOW_Y_TITLE", "FLAG_COUNT_SCOPE",
    "THR_LINE_W", "CONTROL_TINT", "CONTROL_SUFFIX",
    "SHOW_LEGENDS", "LEGEND_SCALE", "LEGEND_SUBGRAPH", "SLIM_LEGEND",
    "DEGREE_BLOCK_EDGE_GREY", "DEGREE_BLOCK_EDGE_ALPHA", "EDGE_WIDTH_SCALE",
    "VIOL_NODE_BOOST", "NODE_SIZE_SCALE", "BASEMAP_ALPHA",
    "DIVIDER", "FONT_S", "FONT_XS", "TITLE_FS", "LEGEND_FS", "TICKNUM_FS", "AXLAB_FS",
    "TITLE_FC", "LEGEND_FC", "TICKNUM_FC", "AXLAB_FC",
    "AXIS_COLOR", "AXIS_WIDTH", "AXIS_ARROWS",
    "SAVE_TRANSPARENT", "SAVE_TIGHT", "SAVE_PAD",
)


# Settings that are read ONCE, before any block is drawn: geometry() runs before the
# loop and the divider after it. Setting one of these per block would either be
# silently ignored or give the two halves inconsistent layouts, so a block that tries
# is rejected rather than quietly misbehaving.
_GLOBAL_ONLY = frozenset({
    "FIG_W_IN", "MAP_H_IN", "MAP_W", "DIST_W", "GAP_MD", "GAP_MID", "MARGIN_IN",
    "FIT_WIDTH", "PX_PER_IN", "DIVIDER", "DIV_SCALE", "DIV_DASH", "DIV_WIDTH",
    "DIV_ALPHA", "GRAPH_SEARCH_DIRS", "GRAPH_CACHE_DIR",
    "SAVE_TRANSPARENT", "SAVE_TIGHT", "SAVE_PAD", "SAVE_PROBE_SCALE",
})


def block_settings(spec: dict) -> dict:
    """
    A block's own settings=, validated.

    Everything drawn inside a block -- its marks, its charts, its key, its crop -- is
    produced under these, so two panels can differ in anything local to them: a denser
    panel can carry a bigger VIOL_NODE_BOOST than its neighbour without touching the
    figure's shared layout.
    """
    kw = spec.get("settings") or {}
    if not kw:
        return {}
    _check(kw)
    bad = sorted(set(kw) & _GLOBAL_ONLY)
    if bad:
        raise KeyError(
            f"{', '.join(bad)} cannot be set per block -- "
            f"{'it is' if len(bad) == 1 else 'they are'} read once for the whole figure "
            f"(geometry before the blocks, the divider after). Pass "
            f"{'it' if len(bad) == 1 else 'them'} to build() / with_blocks() instead.")
    return dict(kw)


def _check(names):
    keys = config_keys()
    unknown = [n for n in names if n not in keys]
    if unknown:
        import difflib
        hint = ""
        for u in unknown:
            near = difflib.get_close_matches(u, keys, n=3, cutoff=0.6)
            if near:
                hint += f"\n  {u} -> did you mean {', '.join(near)}?"
        raise KeyError(f"not a figure2_builder setting: {', '.join(unknown)}.{hint}"
                       f"\n  show_config() lists all {len(keys)} of them. If the name IS "
                       f"in figure2_builder.py, the kernel is running a stale copy -- "
                       f"restart it (autoreload does not re-evaluate module-level data).")


_FONT_ROLES = ("TITLE_FS", "LEGEND_FS", "TICKNUM_FS", "AXLAB_FS")


def _font_fanout(kw: dict) -> dict:
    """FONT_S / FONT_XS propagated to the roles that default to them.

    The whole figure rescales from one keyword, which is the point of having two
    base sizes at all. A role named in the SAME call wins -- setdefault, not
    update -- so FONT_XS=5, TICKNUM_FS=6 means what it reads as. Mirrors
    figure3_builder.configure(), so the same call rescales either figure.
    """
    if "FONT_S" not in kw and "FONT_XS" not in kw:
        return kw
    kw = dict(kw)
    s, xs = kw.get("FONT_S", FONT_S), kw.get("FONT_XS", FONT_XS)
    for role in _FONT_ROLES:
        kw.setdefault(role, s if role == "TITLE_FS" else xs)
    return kw


def configure(**kw):
    """Set settings permanently. Returns the previous values."""
    _check(kw)
    kw = _font_fanout(kw)
    g, before = globals(), {k: globals()[k] for k in kw}
    g.update(kw)
    return before


@contextmanager
def overrides(**kw):
    """Set settings for the duration of a block, then put them back."""
    _check(kw)
    # Fan out BEFORE snapshotting, or a block that sets FONT_XS would leave the
    # four roles it moved still moved once the block is done.
    kw = _font_fanout(kw)
    g, before = globals(), {k: globals()[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(before)


def show_config(common: bool = False):
    """Print the current settings. common=True shows only the ones worth touching."""
    all_keys = config_keys()
    keys = COMMON_KEYS if common else sorted(all_keys)
    width = max(len(k) for k in keys)
    print(f"{len(keys)} of {len(all_keys)} settings" if common
          else f"{len(all_keys)} settings")
    for k in keys:
        v = globals()[k]
        s = repr(v)
        print(f"  {k:<{width}} = {s if len(s) <= 88 else s[:85] + '...'}")


# ═══════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def rgba(hex_color: str, alpha255: float) -> str:
    """UI colours carry 0-255 alphas; plotly wants rgba() with 0-1."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha255 / 255:.3f})"


def mix_white(hex_color: str, amount: float) -> str:
    """
    Fade a colour toward white. amount=1 leaves it alone, 0 erases it.

    Used for the basemap rather than a real alpha: the land, ocean and coastline
    fills stack on each other, so compositing them would darken every overlap.
    Mixing toward the white paper gives the same lightening with no seams.
    """
    h = hex_color.lstrip("#")
    a = min(1.0, max(0.0, amount))
    return "#" + "".join(f"{round(255 + (int(h[i:i + 2], 16) - 255) * a):02x}" for i in (0, 2, 4))


def grey_of(hex_color: str, alpha255: float) -> str:
    """GNNExplorer.hexToGrayRgba — luminance collapse used for de-emphasis."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    lum = round(0.299 * r + 0.587 * g + 0.114 * b)
    return f"rgba({lum},{lum},{lum},{alpha255 / 255:.3f})"


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION, GRAPH LOADING, CROPPING
# ═══════════════════════════════════════════════════════════════════════════

def merc_y(lat):
    """Web-Mercator y in degrees — the projection the UI map uses."""
    lat = np.clip(np.asarray(lat, dtype=float), -MERC_MAX, MERC_MAX)
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def resolve_graph_dir(name) -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for base in GRAPH_SEARCH_DIRS:
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(f"graph '{name}' not found under {[str(b) for b in GRAPH_SEARCH_DIRS]}")


def available_graphs() -> list[str]:
    """Every folder under the search dirs that looks like a graph."""
    seen, out = set(), []
    for base in GRAPH_SEARCH_DIRS:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "__")) or d.name in seen:
                continue
            files = {f.name for f in d.iterdir() if f.is_file()}
            if "grid_lat_lon.pt" in files or is_bsf_format(files):
                seen.add(d.name)
                out.append(d.name)
    return out


def localize_registry(raw: dict) -> list[str]:
    """
    Rebase any subgraph whose edge index is not local to its own node set.

    Most graphs store mesh_up / mesh_down indices per level, which is what the
    registry assumes. Some ocean cluster graphs instead store them in a combined
    two-level space, so one row starts at the size of the finer level and runs
    past the end of the coarser level's coordinate table. That is detectable
    without guessing: a row is only rebased when it actually overruns its node
    count, and only if subtracting the row minimum brings it back into range.
    Returns the keys it repaired, so a surprising graph says so out loud.
    """
    fixed, counts = [], raw["meta"]["node_counts"]
    for key, entry in raw["registry"].items():
        ei, changed = entry["edge_index"], False
        rows = [ei[0].clone(), ei[1].clone()]
        for r, side in ((0, "src"), (1, "tgt")):
            n, row = counts[entry[side]], rows[r]
            if int(row.max()) < n:
                continue
            shifted = row - int(row.min())
            if int(shifted.max()) < n:
                rows[r], changed = shifted, True
            else:
                raise ValueError(
                    f"subgraph '{key}' {side} index reaches {int(row.max())} but "
                    f"'{entry[side]}' has only {n} nodes; rebasing does not fix it."
                )
        if changed:
            entry["edge_index"] = torch.stack(rows)
            fixed.append(key)
    return fixed


def open_graph(name, projection: dict | None = None) -> dict:
    """
    load_graph() for any graph the tool accepts.

    Neural-lam folders load directly. Ocean folders (Njord / baltic-sea-forecasting)
    carry no position files, so they go through the same converter the upload route
    uses; the result is cached under GRAPH_CACHE_DIR and reused.
    """
    src = resolve_graph_dir(name)
    files = {f.name for f in src.iterdir() if f.is_file()}
    if is_bsf_format(files):
        out = GRAPH_CACHE_DIR / src.name
        if not (out / "grid_lat_lon.pt").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"converting ocean graph '{src.name}' → {out} (once)…")
            info = convert_bsf_to_neural_lam(src, out, projection=projection)
            print(f"  {info['format']} · {info['levels']} level(s) · "
                  f"{info['grid_nodes']:,} grid nodes · flavour {info['flavour']}")
        src = out
    raw = load_graph(src)
    if not raw.get("_localized"):
        fixed = localize_registry(raw)
        raw["_localized"] = True
        if fixed:
            print(f"'{name}': rebased combined-space edge indices for {fixed}")
    return raw


def crop_shift(v) -> tuple[float, float]:
    """
    A block's `shift=(dx, dy)`, validated, in CROP WIDTHS and CROP HEIGHTS.

    The unit is the panel itself, not degrees: dx=0.5 slides the view half a panel
    width right, dy=-0.25 a quarter of a panel height down. That keeps a nudge
    meaning the same thing at every zoom and every latitude -- a shift chosen at
    zoom=1 does not fly off the panel when the crop is tightened, and dx does not
    silently mean less near the pole where a longitude degree is shorter.
    """
    if v is None:
        return (0.0, 0.0)
    try:
        dx, dy = v
    except (TypeError, ValueError):
        raise ValueError(f"shift must be (dx, dy) in crop widths / heights, got {v!r}")
    return (float(dx), float(dy))


def crop_bounds(crop: dict) -> dict:
    """
    (lat, lon) window for one panel.

    The height is given in latitude degrees; the width follows from the panel's
    aspect *in Mercator space*, so a cell stays square exactly as it does in the UI.

    `shift` then slides that window off the catalogue centre -- +x right, +y up --
    in units of the window's own width and height, so the crop keeps its size and
    its square cells and only moves. The longitude offset is measured against the
    width the shifted latitudes give, so the panel stays exactly as wide as it
    would have been unshifted.
    """
    lat0, lon0 = crop["center"]
    dx, dy = crop_shift(crop.get("shift"))
    half_h = crop["height"] / 2
    lat0 = lat0 + dy * crop["height"]
    lat_lo, lat_hi = np.clip([lat0 - half_h, lat0 + half_h], -MERC_MAX, MERC_MAX)
    half_w = (merc_y(lat_hi) - merc_y(lat_lo)) / 2 * crop["aspect"]
    lon0 = lon0 + dx * 2 * half_w
    return dict(lat=(float(lat_lo), float(lat_hi)),
                lon=(float(lon0 - half_w), float(lon0 + half_w)),
                center=(float(lat0), float(lon0)))


def _lon_in(lon, lo, hi):
    """bbox_torch._lon_mask — longitude compared modulo 360."""
    lon = np.asarray(lon, dtype=float)
    if hi - lo >= 360:
        return np.ones_like(lon, dtype=bool)
    shift = math.floor((lo + 180) / 360) * 360
    lo, hi = lo - shift, hi - shift
    if hi <= 180:
        return (lon >= lo) & (lon <= hi)
    return (lon >= lo) | (lon <= hi - 360)


def node_mask(lat_lon: np.ndarray, b: dict) -> np.ndarray:
    lat, lon = lat_lon[:, 0], lat_lon[:, 1]
    return (lat >= b["lat"][0]) & (lat <= b["lat"][1]) & _lon_in(lon, *b["lon"])


# ═══════════════════════════════════════════════════════════════════════════
# ONE ROW'S DATA
# ═══════════════════════════════════════════════════════════════════════════

def node_sets(raw: dict) -> dict[str, np.ndarray]:
    """{node_type: [N,2] lat/lon} for every node type in the graph."""
    out = {"grid": raw["grid_lat_lon"].numpy()}
    mesh = raw["mesh_lat_lon"]
    if raw["meta"]["is_hierarchical"]:
        for i, m in enumerate(mesh):
            out[f"mesh_{i}"] = m.numpy()
    else:
        out["mesh"] = mesh.numpy()
    return out


def explore_keys(raw: dict) -> list[str]:
    keys = list(raw["registry"].keys())
    if EXPLORE_SUBGRAPHS == "all":
        return keys
    return [k for k in EXPLORE_SUBGRAPHS if k in keys]


def edge_payload(raw: dict, folder, key: str, b: dict, nodes: dict) -> dict:
    """Cropped edges for one subgraph, with endpoint coordinates and lengths (km)."""
    entry = raw["registry"][key]
    m     = get_graph_metrics(key, folder)
    ei    = entry["edge_index"].numpy()
    src_ll, tgt_ll = nodes[entry["src"]], nodes[entry["tgt"]]
    smask, tmask   = node_mask(src_ll, b), node_mask(tgt_ll, b)
    keep = smask[ei[0]] | tmask[ei[1]]          # bbox_torch: OR semantics
    ei   = ei[:, keep]
    return dict(key=key, entry=entry,
                src_lat=src_ll[ei[0], 0], src_lon=src_ll[ei[0], 1],
                tgt_lat=tgt_ll[ei[1], 0], tgt_lon=tgt_ll[ei[1], 1],
                length=m["lengths"].numpy()[keep])


def build_row(spec: dict) -> dict:
    """Everything one row of the figure needs, cropped and thresholded."""
    folder = resolve_graph_dir(spec["graph"])
    if is_bsf_format({f.name for f in folder.iterdir() if f.is_file()}):
        folder = GRAPH_CACHE_DIR / folder.name
    raw   = open_graph(spec["graph"])
    meta  = raw["meta"]
    key   = spec["subgraph"]
    if key not in raw["registry"]:
        raise KeyError(f"'{key}' not in {sorted(raw['registry'])}")
    entry = raw["registry"][key]
    nodes = node_sets(raw)
    b     = crop_bounds(spec.get("crop") or CROP)

    metrics = get_graph_metrics(key, folder)
    src_deg = metrics["src_deg"].numpy()
    tgt_deg = metrics["tgt_deg"].numpy()
    lengths = metrics["lengths"].numpy()

    masks = {t: node_mask(ll, b) for t, ll in nodes.items()}
    return dict(
        spec=spec, folder=folder, raw=raw, meta=meta, key=key, entry=entry,
        label=spec.get("label") or spec["graph"],
        nodes=nodes, masks=masks, bounds=b,
        src_deg=src_deg, tgt_deg=tgt_deg, lengths=lengths,
        explore_edges=[edge_payload(raw, folder, k, b, nodes) for k in explore_keys(raw)],
        inspect_edges=edge_payload(raw, folder, key, b, nodes),
        n_levels=meta["n_levels"],
        spacing=meta["grid_spacing_deg"],
    )


def resolve_metric(spec: dict, entry: dict) -> str:
    """
    Turn a node-type metric into the concrete side it means for THIS subgraph.

    g2m runs grid -> mesh, so a grid degree is the source out-degree; m2g runs
    mesh -> grid, so the same question is the target in-degree. Naming the node
    type instead of the side is what lets a block switch between g2m and m2g
    without any other edit. "length", "src_deg" and "tgt_deg" pass through.
    """
    metric = spec["metric"]
    if metric not in ("grid_deg", "mesh_deg"):
        return metric
    want = "grid" if metric == "grid_deg" else "mesh"
    hit  = (lambda t: t == "grid") if want == "grid" else (lambda t: t.startswith("mesh"))
    if hit(entry["src"]):
        return "src_deg"
    if hit(entry["tgt"]):
        return "tgt_deg"
    raise KeyError(f"metric '{metric}' needs a '{want}' side, but this subgraph runs "
                   f"{entry['src']} -> {entry['tgt']}")


def node_type_label(node_type: str) -> str:
    """
    The node type as a reader names it: 'grid' -> "grid point", any mesh level
    ('mesh', 'mesh_0', 'mesh_1', ...) -> "mesh node".

    The registry's own names are the graph's internal ones and carry a level index
    the panel does not threshold on, so a chart titled "mesh_0 degree" would be
    naming a thing the reader has no reason to distinguish from mesh_1 here.
    """
    return "grid point" if node_type == "grid" else "mesh node"


def edge_type_label(key: str) -> str:
    """
    The subgraph as a reader names it: any m2m level -> "mesh", g2m and m2g as they
    are -- those two ARE the reader's names for the encoder and decoder edges.
    """
    return "mesh" if key.startswith("m2m") else key


def title_case(text: str) -> str:
    """
    A generated panel title as it is set: capitalised, the rest of it left alone.

    A leading subgraph name goes up whole -- g2m and m2g are acronyms, and
    "G2m edge length" would name a thing the graph does not have.
    """
    head = text.split(" ", 1)[0]
    if head.lower() in ("g2m", "m2g", "m2m"):
        return head.upper() + text[len(head):]
    return text[:1].upper() + text[1:]


def node_glyph(node_type: str) -> str:
    """
    The mark the map draws this node type with: grid -> square, mesh -> circle.

    map_traces picks the symbol from the same test, and the key has to repeat it --
    a legend row about mesh nodes set against a square points at a mark the panel
    never drew.
    """
    return "square" if node_type == "grid" else "circle"


def edge_glyph(key: str) -> str:
    """
    The mark the key draws this subgraph with: m2m -> plain line, everything else
    -> arrow.

    m2m is undirected, so a head on it would assert a direction the graph does not
    have. g2m and m2g ARE the direction -- grid into mesh, mesh back out to grid --
    and the mesh_up / mesh_down levels are directed for the same reason.
    """
    return "line" if key.startswith("m2m") else "arrow"


def _num(v: float) -> str:
    """A threshold as it should be printed: 200.0 -> "200", 265.5 -> "265.5"."""
    return f"{v:g}"


def threshold_phrase(metric: str, node_type: str, edge_key: str,
                     lo, hi) -> str:
    """
    One claim, in the metric's own terms: what this block flagged and where it cut.

    "grid point degree = 0" | "mesh node degree <= 1" | "mesh edge >= 200 km". A
    length row says "edge", not "edge length": the unit carries the metric, and the
    chart beside it is already titled "<edge type> edge length".
    viol_color flags v <= lo and v >= hi, so a degree cut at lo=0 is exactly the
    degree-0 nodes and says so rather than the weaker "<= 0". lo wins when a block
    sets both, matching the one-row-per-claim key.
    """
    low  = lo is not None
    v    = float(lo if low else hi)
    sign = "\u2264" if low else "\u2265"
    if metric == "length":
        return f"{edge_type_label(edge_key)} edge {sign} {_num(v)} km"
    noun = f"{node_type_label(node_type)} degree"
    if low and v == 0:
        return f"{noun} = 0"
    return f"{noun} {sign} {_num(v)}"


def metric_bounds(arr: np.ndarray, integer: bool) -> tuple[float, float]:
    """GNNExplorer.degBoundsGlobal — the padded full range the sliders span."""
    if arr is None or not len(arr):
        return (0.0, 30.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    pad = max(1.0, round(0.05 * (hi - lo)))
    lo, hi = max(0.0, lo - pad), hi + pad
    return (math.floor(lo), math.ceil(hi)) if integer else (lo, hi)


def control_band(hi: float, hi_ctrl) -> tuple[float, float] | None:
    """
    (band_lo, band_hi): the values the two cuts disagree about, or None with no control.

    Ordered rather than assumed. A control set BELOW hi is the looser alternative and
    the band is what it would add; set above, it is the stricter one and the band is
    what it would drop. Either way the band is the disagreement between the two cuts,
    which is the entire subject of a control panel. Half-open at the top, so a mark
    exactly on the outer cut counts as flagged by both.
    """
    if hi_ctrl is None:
        return None
    return (min(hi, hi_ctrl), max(hi, hi_ctrl))


def control_color(base: str = VIOL_ABOVE) -> str:
    """
    The control band's colour: the flag colour it stands beside, mixed toward white.

    Not a new colour. The band is the same violation the block is already flagging,
    seen under a different cut, and a second hue would read as a second KIND of
    defect. Lightness carries "the same claim, weaker evidence for it".
    """
    return mix_white(base, CONTROL_TINT)


def viol_color(v, lo, hi, active: bool, neutral: str = VIOL_NEUTRAL, hi_ctrl=None) -> str:
    if not active:
        return CHART_GREY
    if v <= lo:
        return VIOL_BELOW
    band = control_band(hi, hi_ctrl)
    # Before the hi test, because with a control set above hi the band sits ON the
    # flagged side and would otherwise be swallowed by it.
    if band is not None and band[0] <= v < band[1]:
        return control_color()
    if v >= hi or (band is not None and v >= band[1]):
        return VIOL_ABOVE
    return neutral


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS -- the region catalogue and the failure-mode table
# ═══════════════════════════════════════════════════════════════════════════

def regions_of(cls: str) -> list[str]:
    return [k for k, v in REGIONS.items() if v["cls"] == cls]


def metric_report(graph: str, subgraph: str, metric: str = "length"):
    """
    Print the distribution a threshold has to be chosen against.

    A threshold only means something relative to its own metric's spread, and that
    spread changes a lot between mesh levels -- m2m_0 tops out near 563 km while
    m2m_2 reaches 1,333 km. Run this before setting `hi` or `lo` on a block.
    `metric` accepts "grid_deg" / "mesh_deg" as well as the explicit sides.
    """
    folder = resolve_graph_dir(graph)
    if is_bsf_format({f.name for f in folder.iterdir() if f.is_file()}):
        open_graph(graph); folder = GRAPH_CACHE_DIR / folder.name
    raw = open_graph(graph)
    name = resolve_metric(dict(metric=metric), raw["registry"][subgraph])
    m = get_graph_metrics(subgraph, folder)
    v = (m["lengths"] if name == "length" else
         m["src_deg"] if name == "src_deg" else m["tgt_deg"]).numpy()
    unit = "km" if name == "length" else ""
    side = "" if name == "length" else f" ({name})"
    print(f"{graph} · {subgraph} · {metric}{side}: {v.size:,} values, "
          f"{v.min():.0f}-{v.max():.0f}{unit}")
    if name == "length":
        qs = np.percentile(v, [50, 90, 99, 99.9])
        print(f"   p50 {qs[0]:.0f} | p90 {qs[1]:.0f} | p99 {qs[2]:.0f} | p99.9 {qs[3]:.0f} {unit}")
        for t in (150, 250, 400, 500, 700, 900):
            n = int((v >= t).sum())
            if n:
                print(f"   hi={t:<5} flags {n:>6,} directed ({n // 2:,} undirected)")
    else:
        u, c = np.unique(v, return_counts=True)
        for uu, cc in zip(u[:8], c[:8]):
            print(f"   degree {int(uu)}: {cc:>8,}   (lo={int(uu)} flags {int((v <= uu).sum()):,})")


def grid_isolation(graph: str) -> dict[str, np.ndarray]:
    """{"g2m_iso": bool[N_grid], "m2g_iso": bool[N_grid]} — degree 0 on each side."""
    raw, folder = open_graph(graph), resolve_graph_dir(graph)
    if is_bsf_format({f.name for f in folder.iterdir() if f.is_file()}):
        folder = GRAPH_CACHE_DIR / folder.name
    out = {}
    for key, mode in (("g2m", "g2m_iso"), ("m2g", "m2g_iso")):
        if key not in raw["registry"]:
            continue
        entry = raw["registry"][key]
        side  = resolve_metric(dict(metric="grid_deg"), entry)
        out[mode] = get_graph_metrics(key, folder)[side].numpy() == 0
    return out


def region_counts(aspect: float) -> dict[str, dict]:
    """Per region: undirected long edges per mesh level, and isolated grid nodes."""
    iso_graph = MODE_GRAPHS["g2m_iso"]
    iso   = grid_isolation(iso_graph)
    grid  = open_graph(iso_graph)["grid_lat_lon"].numpy()

    len_graph = MODE_GRAPHS["length"]
    raw_len   = open_graph(len_graph)
    folder    = resolve_graph_dir(len_graph)
    if is_bsf_format({f.name for f in folder.iterdir() if f.is_file()}):
        folder = GRAPH_CACHE_DIR / folder.name
    mesh = node_sets(raw_len)
    levels = {}
    for key, thr in LONG_EDGE_KM.items():
        if key not in raw_len["registry"]:
            continue
        m = get_graph_metrics(key, folder)
        levels[key] = (m["edge_index"].numpy(), m["lengths"].numpy(),
                       mesh[raw_len["registry"][key]["src"]], thr)

    out = {}
    for name, r in REGIONS.items():
        b = crop_bounds(dict(center=r["center"], height=r["height"], aspect=aspect))
        gm = node_mask(grid, b)
        row = dict(cls=r["cls"],
                   nodes={mode: int((m & gm).sum()) for mode, m in iso.items()},
                   edges={})
        for key, (ei, lengths, ll, thr) in levels.items():
            mm   = node_mask(ll, b)
            keep = mm[ei[0]] | mm[ei[1]]
            row["edges"][key] = int(((lengths >= thr) & keep).sum()) // 2
        out[name] = row
    return out


def region_table(aspect: float = None):
    """The region x failure-mode matrix, printed."""
    aspect = aspect if aspect is not None else (geometry()["map_px"] / geometry()["row_px"])
    counts = region_counts(aspect)
    lv = [k for k in LONG_EDGE_KM if any(k in c["edges"] for c in counts.values())]

    print("region × failure mode   (panel aspect "
          f"{aspect:.3f}, crop heights from §3, ZOOM not applied)")
    print(f"  L  long m2m edges, undirected, on {MODE_GRAPHS['length']} · "
          + " · ".join(f"{k} ≥ {int(LONG_EDGE_KM[k])} km" for k in lv))
    print(f"  G  grid nodes with no g2m edge, M  with no m2g edge, "
          f"on {MODE_GRAPHS['g2m_iso']}")
    print(f"  a letter needs ≥ {MODE_MIN_EDGES} edge / ≥ {MODE_MIN_NODES} nodes in the crop\n")

    head = (f"{'region':<17} {'class':<8}" + "".join(f"{k.replace('m2m_', 'L'):>6}" for k in lv)
            + f"{'g2m':>7}{'m2g':>6}   modes")
    print(head); print("─" * len(head))

    shown = None
    for name, c in counts.items():
        if shown is not None and c["cls"] != shown:
            print()
        shown = c["cls"]
        n_edges = sum(c["edges"].values())
        flags = "".join((
            "L" if n_edges >= MODE_MIN_EDGES else "·",
            "G" if c["nodes"].get("g2m_iso", 0) >= MODE_MIN_NODES else "·",
            "M" if c["nodes"].get("m2g_iso", 0) >= MODE_MIN_NODES else "·"))
        print(f"{name:<17} {c['cls']:<8}"
              + "".join(f"{c['edges'][k]:>6,}" for k in lv)
              + f"{c['nodes'].get('g2m_iso', 0):>7,}{c['nodes'].get('m2g_iso', 0):>6,}"
              + f"   {' '.join(flags)}")

    # "other" holds whole-graph views -- global would win every mode by being the
    # whole world, which says nothing about where to point a panel.
    panels = {k: v for k, v in counts.items() if v["cls"] != "other"}
    print("\nrichest crop per mode (whole-graph views excluded):")
    for mode, pick in (("L", lambda c: sum(c["edges"].values())),
                       ("G", lambda c: c["nodes"].get("g2m_iso", 0)),
                       ("M", lambda c: c["nodes"].get("m2g_iso", 0))):
        best = sorted(panels.items(), key=lambda kv: -pick(kv[1]))[:3]
        print(f"  {mode}  " + " · ".join(f"{k} ({pick(v):,})" for k, v in best))


def print_regions():
    """The crop catalogue, grouped by the failure mode each crop was found for."""
    for cls, blurb in CLASSES.items():
        keys = regions_of(cls)
        print(f"\n{cls.upper():<9} {blurb}   [{len(keys)}]")
        for k in keys:
            v = REGIONS[k]
            print(f"   {k:<17} ({v['center'][0]:6.1f}, {v['center'][1]:7.1f})  "
                  f"h={v['height']:5.1f}deg  {v['note']}")


# ═══════════════════════════════════════════════════════════════════════════
# MAP MARKS -- constants from GNNExplorer.tsx / LeftPanel.tsx
# ═══════════════════════════════════════════════════════════════════════════

def node_level_boost(node_type: str, n_levels: int) -> float:
    """LeftPanel.nodeLevelBoost — coarse mesh levels draw larger."""
    if n_levels <= 1 or not node_type.startswith("mesh_"):
        return 1.0
    lvl = int(node_type.split("_")[1])
    return MESH_NODE_FINE_BOOST + (MESH_NODE_COARSE_BOOST - MESH_NODE_FINE_BOOST) * lvl / (n_levels - 1)


def stroke_for_level(lvl: int, n_levels: int) -> float:
    """LeftPanel.strokeForLevel — coarse mesh levels draw heavier."""
    if n_levels <= 1:
        return (MESH_STROKE_COARSE + MESH_STROKE_FINE) / 2
    inv = n_levels - 1 - lvl
    return MESH_STROKE_COARSE - (MESH_STROKE_COARSE - MESH_STROKE_FINE) * inv / (n_levels - 1)


def base_node_px(row: dict, panel_w_px: float) -> float:
    """
    Grid-node size in points, following the UI's world-unit rule.

    GNNExplorer sizes a node at NODE_SIZE_M scaled by the graph's own grid
    spacing, so a node always covers the same fraction of one grid cell. Here
    that world size is converted into the panel's pixels via the crop width,
    which is what makes a 0.25 deg ocean grid read like the UI instead of
    fusing into a solid sheet.
    """
    spacing  = row["spacing"] or REFERENCE_SPACING_DEG
    size_deg = (NODE_SIZE_M / 111_320) * (spacing / REFERENCE_SPACING_DEG)
    lon_span = row["bounds"]["lon"][1] - row["bounds"]["lon"][0]
    px = size_deg / lon_span * panel_w_px * NODE_SIZE_SCALE
    return float(np.clip(px, NODE_SIZE_MIN_PX, NODE_SIZE_MAX_PX))


def mesh_node_px(row: dict, base_px: float, node_type: str,
                 full_view: bool, is_endpoint: bool) -> float:
    """
    Size for a mesh node, in points.

    The UI's rule (base size x MESH_SIZE_SCALE x the level boost) assumes marks
    calibrated to the mesh. Here base_px comes from base_node_px(), which is
    calibrated to the GRID, so a fine mesh level lands far below a pixel: on the
    0.25 deg ocean grid, mesh_0 comes out at 0.27 px -- drawn, but not on the page.

    The floor applies only to a COUNTERPART endpoint: the mesh side of a degree
    block, which is the thing the flagged nodes are missing edges to, and which is
    useless at sub-pixel size. A length block's own mesh nodes keep the UI's rule
    untouched, so nothing about that panel moves. In full view the floor is raised
    again, because there the endpoints are the subject rather than the reference.
    """
    px = base_px * MESH_SIZE_SCALE * node_level_boost(node_type, row["n_levels"])
    if not is_endpoint:
        return float(px)
    return float(max(px, MESH_NODE_MIN_PX * (1.6 if full_view else 1.0)))


def arrow_trace(geo, payload, mask, color, size_px):
    """
    Arrowheads at the target end of a directed subgraph.

    The UI draws these as rotated sprites; here they are rotated triangle
    markers. The rotation is the edge's bearing in Mercator space, which is the
    space the panel is drawn in, so an arrow lines up with its own edge.

    The caller decides whether arrows are wanted; this only refuses an empty mask.
    """
    if not mask.any():
        return None
    s_lat, s_lon = payload["src_lat"][mask], payload["src_lon"][mask]
    t_lat, t_lon = payload["tgt_lat"][mask], payload["tgt_lon"][mask]
    keep = np.abs(t_lon - s_lon) <= 180
    if not keep.any():
        return None
    s_lat, s_lon, t_lat, t_lon = s_lat[keep], s_lon[keep], t_lat[keep], t_lon[keep]
    theta = np.degrees(np.arctan2(merc_y(t_lat) - merc_y(s_lat), t_lon - s_lon))
    return go.Scattergeo(
        lat=t_lat, lon=t_lon, mode="markers", geo=geo,
        marker=dict(symbol="triangle-up", size=size_px, color=color,
                    angle=90 - theta, angleref="up", line=dict(width=0)),
        hoverinfo="skip", showlegend=False)


def _segments(src_lat, src_lon, tgt_lat, tgt_lon):
    """lat/lon arrays with a None after every segment; antimeridian jumps dropped."""
    keep = np.abs(tgt_lon - src_lon) <= 180
    n = int(keep.sum())
    if n == 0:
        return [], []
    nan = np.full(n, np.nan)
    lon = np.ravel(np.column_stack([src_lon[keep], tgt_lon[keep], nan]))
    lat = np.ravel(np.column_stack([src_lat[keep], tgt_lat[keep], nan]))
    return lat, lon


def edge_trace(geo, payload, mask, color, width, name=None):
    lat, lon = _segments(payload["src_lat"][mask], payload["src_lon"][mask],
                         payload["tgt_lat"][mask], payload["tgt_lon"][mask])
    if len(lat) == 0:
        return None
    return go.Scattergeo(lat=lat, lon=lon, mode="lines", geo=geo, name=name or "",
                         line=dict(color=color, width=width),
                         hoverinfo="skip", showlegend=False)


def node_trace(geo, lat_lon, mask, symbol, size_px, color, border_px):
    if not mask.any():
        return None
    ll = lat_lon[mask]
    return go.Scattergeo(
        lat=ll[:, 0], lon=ll[:, 1], mode="markers", geo=geo,
        marker=dict(symbol=symbol, size=size_px, color=color,
                    line=dict(color=rgba(TEXT, NODE_ICON_ALPHA), width=border_px)),
        hoverinfo="skip", showlegend=False)


def _lon_spans(bounds: dict) -> list[tuple[float, float, float]]:
    """
    (lo, hi, shift) windows in Natural Earth's own -180..180 frame.

    A crop may run past the antimeridian -- crop_bounds does not wrap, because the
    panel's lonaxis wants a continuous range. The shapefiles do not wrap either, so
    a straddling crop is clipped as two windows and the second is shifted back by
    360 to sit where the panel expects it.
    """
    lo, hi = bounds["lon"]
    if hi - lo >= 360:
        return [(-180.0, 180.0, 0.0)]
    span = hi - lo
    lo_n = ((lo + 180.0) % 360.0) - 180.0
    hi_n = lo_n + span
    shift = lo - lo_n
    if hi_n <= 180.0:
        return [(lo_n, hi_n, shift)]
    return [(lo_n, 180.0, shift), (-180.0, hi_n - 360.0, shift + 360.0)]


@lru_cache(maxsize=64)
def _ne_rings(res: str, name: str, lo: float, la0: float, hi: float, la1: float,
              shift: float) -> tuple:
    """Clipped polygon rings for one Natural Earth layer, as (lat, lon) tuples."""
    from cartopy.io import shapereader
    from shapely.geometry import box

    path = shapereader.natural_earth(resolution=res, category="physical", name=name)
    clip = box(lo, la0, hi, la1)
    lat: list = []
    lon: list = []
    for g in shapereader.Reader(path).geometries():
        if not g.is_valid:
            g = g.buffer(0)
        if not g.intersects(clip):
            continue
        gi = g.intersection(clip)
        parts = gi.geoms if gi.geom_type.startswith("Multi") else [gi]
        for poly in parts:
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            # Winding order is load-bearing, not cosmetic. On a Scattergeo, fill="toself"
            # fills the side of the ring its direction implies, so a ring wound the wrong
            # way fills the whole rest of the globe instead of its own interior -- one
            # 0.2-degree lake island painted over an entire panel this way.
            #
            # Plotly has no hole support, so every ring is drawn as its own filled shape,
            # which means EVERY ring has to wind the same way -- including interiors.
            # Orienting the polygon is not enough and is actively wrong here: it winds
            # interiors opposite to exteriors by definition, which is exactly the case
            # that floods. So each ring is normalised to clockwise on its own, by the
            # sign of its shoelace area. A lake inside land is therefore painted as land
            # and then painted back by the lakes layer, which draws second.
            for ring in [poly.exterior, *poly.interiors]:
                x, y = ring.xy
                rx, ry = np.asarray(x, float), np.asarray(y, float)
                if 0.5 * np.sum(rx[:-1] * ry[1:] - rx[1:] * ry[:-1]) > 0:
                    rx, ry = rx[::-1], ry[::-1]
                lon += list(rx + shift) + [None]
                lat += list(ry) + [None]
    return tuple(lat), tuple(lon)


@lru_cache(maxsize=64)
def _ne_lines(res: str, name: str, lo: float, la0: float, hi: float, la1: float,
              shift: float) -> tuple:
    """Clipped LINE features (coastline), as (lat, lon) tuples."""
    from cartopy.io import shapereader
    from shapely.geometry import box

    path = shapereader.natural_earth(resolution=res, category="physical", name=name)
    clip = box(lo, la0, hi, la1)
    lat: list = []
    lon: list = []
    for g in shapereader.Reader(path).geometries():
        if not g.intersects(clip):
            continue
        gi = g.intersection(clip)
        parts = gi.geoms if gi.geom_type.startswith(("Multi", "Geometry")) else [gi]
        for line in parts:
            if line.is_empty or line.geom_type != "LineString":
                continue
            x, y = line.xy
            lon += [v + shift for v in x] + [None]
            lat += list(y) + [None]
    return tuple(lat), tuple(lon)


def ne_basemap_traces(row: dict, geo: str) -> list:
    """
    Land and lakes at BASEMAP_NE_RES, clipped to the crop, drawn under everything.

    Fills and strokes come from different datasets on purpose. Filling from the
    clipped POLYGONS and stroking the same rings would draw the clip box's own edges
    as if they were coastline -- a false shoreline straight across the panel wherever
    a landmass runs out of the frame. So the polygons are filled with no outline and
    the shoreline is stroked from the coastline dataset, whose clipped LineStrings
    contain only real coast.

    Colours are the same GEO ones plotly would have used, through the same
    mix_white() tint, so switching source changes the detail and nothing else.
    """
    tint = lambda c: mix_white(c, BASEMAP_ALPHA)
    spans = _lon_spans(row["bounds"])
    la0, la1 = row["bounds"]["lat"]
    out = []
    for name, color in (("land", tint(GEO["landcolor"])), ("lakes", tint(GEO["lakecolor"]))):
        lat: list = []
        lon: list = []
        for lo, hi, shift in spans:
            a, b = _ne_rings(BASEMAP_NE_RES, name, lo, la0, hi, la1, shift)
            lat += list(a)
            lon += list(b)
        if lat:
            out.append(go.Scattergeo(
                lat=lat, lon=lon, mode="lines", geo=geo, fill="toself", fillcolor=color,
                line=dict(width=0), hoverinfo="skip", showlegend=False))

    lat, lon = [], []
    for lo, hi, shift in spans:
        a, b = _ne_lines(BASEMAP_NE_RES, "coastline", lo, la0, hi, la1, shift)
        lat += list(a)
        lon += list(b)
    if lat:
        out.append(go.Scattergeo(
            lat=lat, lon=lon, mode="lines", geo=geo,
            line=dict(color=tint(GEO["coastlinecolor"]), width=GEO["coastlinewidth"]),
            hoverinfo="skip", showlegend=False))
    return out


def geo_layout(row: dict) -> dict:
    """One geo subplot: Mercator, cropped, desaturated basemap."""
    b = row["bounds"]
    tint = lambda c: mix_white(c, BASEMAP_ALPHA)   # frame excluded — panel chrome, not map
    # When Natural Earth draws the land, plotly must not draw it too: two coastlines
    # from different generalisations diverge by kilometres at these crop sizes.
    ne = BASEMAP_SOURCE == "naturalearth"
    return dict(
        projection=dict(type="mercator"),
        lataxis=dict(range=list(b["lat"])),
        lonaxis=dict(range=list(b["lon"])),
        showland=not ne, showocean=True, showlakes=not ne, showcoastlines=not ne,
        showframe=True, showrivers=False,
        landcolor=tint(GEO["landcolor"]), oceancolor=tint(GEO["oceancolor"]),
        lakecolor=tint(GEO["lakecolor"]),
        coastlinecolor=tint(GEO["coastlinecolor"]), coastlinewidth=GEO["coastlinewidth"],
        showcountries=GEO["showcountries"], countrycolor=tint(GEO["countrycolor"]),
        countrywidth=GEO["countrywidth"], resolution=GEO["resolution"],
        framecolor=GEO["framecolor"], framewidth=GEO["framewidth"],
        # Ocean colour, NOT transparent. plotly's ocean layer does not reliably cover
        # the whole domain -- it stops at its own polygon edge -- and a transparent
        # background lets the white PAPER through wherever it falls short, which
        # renders as a bright band with a curved top across the bottom of the panel.
        # plotly's land layer happens to hide it in the default path; switch that off
        # for a Natural Earth basemap and it is plainly visible. Painting the geo
        # background the ocean colour means any such gap falls through to ocean.
        bgcolor=tint(GEO["oceancolor"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHART PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def cluster_by_gap(values: np.ndarray, gap_km: float = None) -> list[tuple[float, int]]:
    """GNNExplorer.clusterByGap — collapse near-identical lengths onto their median."""
    gap_km = LEN_CLUSTER_GAP if gap_km is None else gap_km
    if not len(values):
        return []
    v = np.sort(np.asarray(values, dtype=float))
    cuts = np.flatnonzero(np.diff(v) > gap_km) + 1
    return [(float(round(np.median(c))), int(len(c))) for c in np.split(v, cuts) if len(c)]


def value_counts(values: np.ndarray) -> list[tuple[float, int]]:
    v, n = np.unique(np.asarray(values), return_counts=True)
    return list(zip(v.astype(float), n.astype(int)))


def lollipop_geometry(entries, domain, panel_px: float) -> tuple[float, tuple]:
    """
    (head diameter in points, padded x range) for one lollipop panel.

    The head is sized from the pitch between adjacent items measured in PIXELS, so a
    crowded chart gets a smaller dot with no per-panel setting -- which is the whole
    point: the two charts of a split column hold different numbers of degrees in the
    same width. It is capped at LOLLI_HEAD_PX, so nothing ever draws larger than the
    old fixed size.

    The left pad is expressed in head diameters, which makes it circular: padding
    widens the range, which tightens the pitch, which shrinks the head, which shortens
    the pad. Three passes settle it -- the movement after the first is already under a
    tenth of a point -- and settling it matters because the axis range this returns is
    the one the head was sized against.
    """
    lo, hi = float(domain[0]), float(domain[1])
    xs = sorted(float(v) for v, _ in entries)
    span0 = (hi - lo) or 1.0
    pitch = min((b - a for a, b in zip(xs, xs[1:])), default=span0)
    head, pad, span = LOLLI_HEAD_PX, 0.0, span0
    for _ in range(3):
        px_per_unit = panel_px / span
        head = min(LOLLI_HEAD_PX, max(LOLLI_HEAD_MIN, pitch * px_per_unit * LOLLI_HEAD_FRAC))
        pad  = LOLLI_X_PAD * head / px_per_unit
        span = span0 + pad
    return head, (lo - pad, hi)


def lollipop_traces(entries, lo, hi, active, axis: tuple[str, str],
                    head_px: float | None = None, hi_ctrl=None) -> list:
    """
    Stem-and-dot frequency plot, exactly the tool's encoding: stems at LOLLI_STEM_W /
    0.8 opacity, hollow dots with a coloured ring LOLLI_HEAD_W wide, coloured by which
    side of the thresholds the value falls on.

    head_px is the dot diameter, normally from lollipop_geometry() so it matches the
    axis range drawn beside it. None falls back to the uncrowded default.
    """
    xa, ya = axis
    if not entries:
        return []
    head_px = LOLLI_HEAD_PX if head_px is None else head_px
    buckets: dict[str, list] = {}
    for v, n in entries:
        buckets.setdefault(viol_color(v, lo, hi, active, hi_ctrl=hi_ctrl), []).append((v, n))
    out = []
    for color, pts in buckets.items():
        xs = np.ravel([[v, v, np.nan] for v, _ in pts])
        ys = np.ravel([[0, n, np.nan] for _, n in pts])
        out.append(go.Scatter(x=xs, y=ys, mode="lines", xaxis=xa, yaxis=ya,
                              line=dict(color=color, width=LOLLI_STEM_W), opacity=0.8,
                              hoverinfo="skip", showlegend=False))
        out.append(go.Scatter(x=[v for v, _ in pts], y=[n for _, n in pts], mode="markers",
                              xaxis=xa, yaxis=ya, opacity=0.9,
                              marker=dict(size=head_px, color="white",
                                          line=dict(color=color, width=LOLLI_HEAD_W)),
                              customdata=[[v, n] for v, n in pts],
                              hovertemplate="%{customdata[0]:g} · n=%{customdata[1]:,}<extra></extra>",
                              showlegend=False))
    return out


def histogram_traces(values, domain, lo, hi, active, n_bins, axis, hi_ctrl=None) -> list:
    """
    Equal-width bins over the metric's full padded range, coloured per side of the
    thresholds.

    A bin the threshold falls INSIDE is cut at the threshold and its two sides are
    drawn in their own colours, rather than the whole bar taking the colour of its
    midpoint. Colouring by midpoint mis-states the bar it is drawn on: a bin
    straddling `hi` is part flagged and part not, and painting all of it one colour
    either invents flagged edges or hides real ones, by up to a bin's worth. Bins
    clear of every threshold are still one bar.

    The cut is placed on the bar's DRAWN extent, not the bin's nominal one, so the
    0.94 inter-bin gap stays at the bin's outer edges and the two halves meet flush.
    Hover reports the bin -- its midpoint and its full count -- on either half,
    because the segments are one bar split for colour, not two measurements.
    """
    xa, ya = axis
    values = np.asarray(values, dtype=float)
    if not len(values):
        return []
    edges = np.linspace(domain[0], domain[1], max(2, n_bins) + 1)
    counts, _ = np.histogram(np.clip(values, domain[0], domain[1]), bins=edges)
    mids  = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.94

    # Every value that changes a mark's colour, so a control band's inner edge cuts
    # the bar it lands in exactly as lo and hi do. Unset thresholds sit ON the padded
    # range edge, never strictly inside a bin, so they cut nothing.
    cuts_all = [c for c in (lo, hi, hi_ctrl) if c is not None] if active else []

    xs, ws, cs, hover = [], [], [], []
    for e0, e1, mid, n in zip(edges[:-1], edges[1:], mids, counts):
        d0, d1 = mid - width / 2, mid + width / 2
        inner  = sorted({float(c) for c in cuts_all if e0 < c < e1})
        bounds = [e0, *inner, e1]
        for a, b in zip(bounds, bounds[1:]):
            # Nominal bin fraction -> drawn extent, so the gap is not reintroduced
            # in the bin's interior.
            pa = d0 + (a - e0) / (e1 - e0) * (d1 - d0)
            pb = d0 + (b - e0) / (e1 - e0) * (d1 - d0)
            xs.append((pa + pb) / 2)
            ws.append(pb - pa)
            cs.append(viol_color((a + b) / 2, lo, hi, active, hi_ctrl=hi_ctrl))
            hover.append([mid, n])

    return [go.Bar(x=xs, y=[h[1] for h in hover], width=ws, xaxis=xa, yaxis=ya,
                   marker=dict(color=cs, line=dict(width=0)), opacity=0.75,
                   customdata=hover,
                   hovertemplate="%{customdata[0]:.0f} · n=%{customdata[1]:,}<extra></extra>",
                   showlegend=False)]


def bar_traces(entries, lo, hi, active, axis, hi_ctrl=None) -> list:
    """NODE_CHART_STYLE = 'bar' — same colours, filled bars instead of stems."""
    xa, ya = axis
    if not entries:
        return []
    xs = [v for v, _ in entries]
    step = min((b - a for a, b in zip(xs, xs[1:])), default=1.0) if len(xs) > 1 else 1.0
    return [go.Bar(x=xs, y=[n for _, n in entries], width=step * 0.7, xaxis=xa, yaxis=ya,
                   marker=dict(color=[viol_color(v, lo, hi, active, hi_ctrl=hi_ctrl)
                                      for v in xs],
                               line=dict(width=0)), opacity=0.75,
                   hovertemplate="%{x:g} · n=%{y:,}<extra></extra>", showlegend=False)]


def threshold_shapes(xa: str, ya: str, info: dict) -> list[dict]:
    """
    The two dashed threshold markers, as shape dicts.

    Returned rather than added, because the threshold chips swap the whole shape
    list at once and the markers have to travel with the colours they explain.
    xa / ya are trace-style axis references, e.g. "x" and "y3".
    """
    t = info["thr"]
    if not t["active"]:
        return []
    # On a discrete metric the marker steps off the value it describes: degrees are
    # integers, so a line drawn exactly on lo runs down the middle of the lollipop it
    # is meant to bound. Half a degree out puts it in the empty space between lo and
    # lo+1 -- which is where the flagged/neutral boundary actually is. The hi marker
    # mirrors it inward, for the same reason.
    off = DEG_THR_OFFSET if info.get("discrete") else 0.0
    x0, x1 = info["x_range"]
    out = []
    markers = [(t["lo"] + off, VIOL_BELOW, t["lo_set"]),
               (t["hi"] - off, VIOL_ABOVE, t["hi_set"])]
    # The control marker travels with the band it bounds: same dash, the lighter shade
    # the band's marks are drawn in, so the two lines and the key's two rows read as
    # the one comparison rather than as two unrelated claims.
    if t.get("hi_ctrl") is not None:
        markers.append((t["hi_ctrl"] - off, control_color(), True))
    for value, color, asked in markers:
        if not asked or not (x0 < value < x1):
            continue          # not this block's claim, or off the drawn range
        out.append(dict(type="line", x0=value, x1=value, y0=0, y1=1,
                        xref=xa, yref=ya + " domain", opacity=0.9, layer="above",
                        line=dict(color=color, width=THR_LINE_W, dash="3px,2px")))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def geometry() -> dict:
    """Panel domains in paper coordinates, plus the pixel sizes the marks need."""
    fig_w = FIG_W_IN * PX_PER_IN
    m = {k: v * PX_PER_IN for k, v in MARGIN_IN.items()}
    plot_w = fig_w - m["l"] - m["r"]
    plot_h = MAP_H_IN * PX_PER_IN
    fig_h  = plot_h + m["t"] + m["b"]

    map_w, dist_w = MAP_W, DIST_W
    used = 2 * (map_w + GAP_MD + dist_w) + GAP_MID + (m["l"] + m["r"]) / fig_w
    if FIT_WIDTH:
        # Absorb the slack into the panels, keeping their ratio to each other.
        slack   = 1.0 - used
        panels  = 2 * (map_w + dist_w)
        scale   = 1.0 + slack / panels
        map_w, dist_w = map_w * scale, dist_w * scale
        used = 1.0
    elif abs(used - 1.0) > 1e-9:
        print(f"note: the row uses {used:.4f} of the width "
              f"({'overflows by' if used > 1 else 'leaves'} {abs(1-used)*FIG_W_IN:.3f} in). "
              f"Set FIT_WIDTH=True to absorb it.")

    # Paper coords run 0..1 across the PLOT area, not the figure, so convert.
    f2p = lambda frac: frac * fig_w / plot_w        # figure fraction -> paper fraction
    mw, dw, gmd, gmid = f2p(map_w), f2p(dist_w), f2p(GAP_MD), f2p(GAP_MID)

    x, cursor = [], 0.0
    for block in range(2):
        x.append((cursor, cursor + mw));            cursor += mw + gmd
        x.append((cursor, cursor + dw));            cursor += dw
        if block == 0:
            cursor += gmid
    x_div = x[1][1] + gmid / 2

    # The cursor is accumulated by addition, so the last edge lands on 1.0 + 1e-16 for
    # most gap values -- and plotly rejects a domain above 1 with a type error that
    # names neither the axis nor the setting. The default gaps happen to sum exactly,
    # which is why this only appeared once GAP_MD / GAP_MID were touched. FIT_WIDTH has
    # already made the widths sum to 1, so any overshoot here is float noise; a real
    # overflow (FIT_WIDTH=False and the row genuinely too wide) is reported, not hidden.
    over = x[-1][1] - 1.0
    if over > 1e-6:
        print(f"note: the row overflows the width by {over * FIG_W_IN:.3f} in; panels "
              f"clipped at the edge. Set FIT_WIDTH=True or shrink the gaps.")
    x = [(min(max(a, 0.0), 1.0), min(max(b, 0.0), 1.0)) for a, b in x]
    x_div = min(max(x_div, 0.0), 1.0)

    return dict(x=x, y=(0.0, 1.0), x_div=x_div,
                fig_w=fig_w, fig_h=fig_h, plot_w=plot_w, plot_h=plot_h, margin=m,
                map_px=mw * plot_w, dist_px=dw * plot_w, row_px=plot_h,
                map_w_frac=map_w, dist_w_frac=dist_w)


def block_crop(spec: dict, dom: dict) -> dict:
    """
    The crop SPEC for one block -- centre, height, and the aspect its own panel
    demands. build_row() turns this into bounds; returning bounds here instead
    would be cropped twice.

    Zoom divides the height: the block's own zoom= times the global ZOOM, so >1
    tightens the view about the same centre. Only the height is touched -- the
    longitude span is derived from it and the panel aspect downstream, so a zoomed
    crop stays exactly as square-celled as an unzoomed one.

    shift=(dx, dy) then moves that view: +dx right, -dx left, +dy up, -dy down, in
    units of the crop's own width and height (dx=0.5 is half a panel right). It is
    applied AFTER the zoom, so it is measured against the crop actually drawn --
    re-zoom a block and its nudge still lands where it did on screen.
    """
    if spec.get("crop"):
        c = dict(spec["crop"])
    else:
        if not spec.get("region"):
            raise KeyError(f"block '{spec.get('title') or spec['graph']}' has neither "
                           f"'region' nor 'crop'")
        r = REGIONS[spec["region"]]
        c = dict(center=r["center"], height=r["height"])
    z = float(spec.get("zoom") or 1.0) * ZOOM
    if z <= 0:
        raise ValueError(f"zoom must be positive, got {z}")
    c["height"] = c["height"] / z
    c["aspect"] = dom["map_px"] / dom["row_px"]
    c["shift"] = crop_shift(spec.get("shift", c.get("shift")))
    return c


def node_metric_of(spec: dict, entry: dict) -> str | None:
    """
    The block's SECOND claim, if it made one.

    A block normally makes one claim, which is what keeps a panel readable. Some
    do not fit that: the North Sea / Baltic crop is about long edges hopping
    Denmark AND about the Baltic mesh nodes those hops leave behind, and splitting
    them across two panels would break the very adjacency that is the point. Such a
    block sets node_metric= (with node_lo / node_hi) alongside its edge metric, and
    both are thresholded in the one map.
    """
    nm = spec.get("node_metric")
    if nm is None:
        return None
    nm = resolve_metric(dict(metric=nm), entry)
    if nm == "length":
        raise ValueError("node_metric must be a degree, not 'length'")
    if nm == resolve_metric(spec, entry):
        raise ValueError(f"node_metric resolves to '{nm}', the same side as metric; "
                         f"drop one of them")
    return nm


def _control_value(spec: dict, hi_key: str, ctrl_key: str, hi_set: bool):
    """
    One claim's control cut, validated.

    A control is a second cut BESIDE the block's own threshold, and the band between
    them is the point; with no `hi` there is nothing for it to sit beside and no band
    to draw, so a lone control reads as a typo and is rejected rather than silently
    becoming the threshold.
    """
    v = spec.get(ctrl_key)
    if v is None:
        return None
    if not hi_set:
        raise ValueError(f"{ctrl_key}={v} needs {hi_key}= on the same claim: a control "
                         f"is a second cut beside the block's threshold, not a "
                         f"replacement for it")
    return float(v)


def block_thresholds(row: dict, spec: dict) -> dict:
    metric = resolve_metric(spec, row["entry"])
    node_m = node_metric_of(spec, row["entry"])
    out = {}
    for name, arr, integer in (("src_deg", row["src_deg"], True),
                               ("tgt_deg", row["tgt_deg"], True),
                               ("length",  row["lengths"], False)):
        full = metric_bounds(arr, integer)
        if name == metric:
            lo = full[0] if spec.get("lo") is None else float(spec["lo"])
            hi = full[1] if spec.get("hi") is None else float(spec["hi"])
            # `active` here means "the block asked for a threshold", not "the value
            # sits inside the padded range". Degrees pad down to 0, so a threshold of
            # lo=0 -- the one that isolates degree-0 nodes, the whole point of the
            # right-hand panel -- would otherwise read as inactive and recolour nothing.
            active = spec.get("lo") is not None or spec.get("hi") is not None
            lo_set, hi_set = spec.get("lo") is not None, spec.get("hi") is not None
            hi_ctrl = _control_value(spec, "hi", "hi_control", hi_set)
        elif name == node_m:
            lo = full[0] if spec.get("node_lo") is None else float(spec["node_lo"])
            hi = full[1] if spec.get("node_hi") is None else float(spec["node_hi"])
            active = spec.get("node_lo") is not None or spec.get("node_hi") is not None
            lo_set = spec.get("node_lo") is not None
            hi_set = spec.get("node_hi") is not None
            hi_ctrl = _control_value(spec, "node_hi", "node_hi_control", hi_set)
        else:
            lo, hi, active = full[0], full[1], False
            lo_set = hi_set = False
            hi_ctrl = None
        # Per side, because a marker is drawn for a side the block ASKED for, not for
        # one that merely sits inside the range. lo=0 on a degree metric is the case
        # that forced this: it is the threshold that isolates degree-0 nodes, and the
        # old "is it at the range edge" test suppressed its marker entirely.
        out[name] = dict(lo=lo, hi=hi, full=full, active=active,
                         lo_set=lo_set, hi_set=hi_set, hi_ctrl=hi_ctrl)
    return out


def _count_for(row: dict, thr: dict, metric: str) -> tuple[int, int]:
    """
    (flagged, total) for one resolved metric, under FLAG_COUNT_SCOPE.

    Edges are reported undirected, matching region_table() and the header contract.
    An m2m subgraph runs mesh -> mesh and stores every edge in both directions, so
    the one long hop over Wales sits in the array twice and draws as two coincident
    lines -- a caption reading "2 edges" under a map showing one line is simply
    wrong. Halving is exact rather than approximate: the crop's OR-semantics window
    (smask[src] | tmask[tgt]) is symmetric in the two endpoints, so both directions
    of an edge are always kept or dropped together. g2m, m2g and the mesh_up /
    mesh_down levels have no reverse edge and are counted as they are stored.
    """
    t = thr[metric]
    if not t["active"]:
        return 0, 0
    if metric == "length":
        v = np.asarray(row["lengths"])
        if FLAG_COUNT_SCOPE == "crop":
            v = np.asarray(row["inspect_edges"]["length"])
    else:
        side = "src" if metric == "src_deg" else "tgt"
        v = np.asarray(row["src_deg"] if side == "src" else row["tgt_deg"])
        if FLAG_COUNT_SCOPE == "crop":
            v = v[row["masks"][row["entry"][side]]]
    n, tot = int(((v <= t["lo"]) | (v >= t["hi"])).sum()), int(v.size)
    entry = row["entry"]
    if metric == "length" and entry["src"] == entry["tgt"]:
        n, tot = n // 2, tot // 2
    return n, tot


def undirected_lengths(row: dict) -> np.ndarray:
    """
    row["lengths"] with an m2m subgraph's duplicate reverse edges dropped.

    Same correction _count_for() makes, for the chart beside the map: a symmetric
    subgraph stores i->j and j->i, so every histogram bar and every lollipop stem
    stands at exactly twice the number of edges the map draws. Taking src < tgt
    keeps one arbitrary direction of each pair, which is the whole undirected set
    and leaves the length distribution's shape untouched.
    """
    entry = row["entry"]
    lengths = np.asarray(row["lengths"])
    if entry["src"] != entry["tgt"]:
        return lengths
    ei = entry["edge_index"].numpy()
    return lengths[ei[0] < ei[1]]


def flagged_parts(row: dict, spec: dict, thr: dict) -> list[tuple[int, int, str]]:
    """
    (flagged, total, unit) per claim the block made -- one entry, or two for a
    dual-threshold block. The caption under the map is built from this, so a panel
    that flags two things says so instead of quietly reporting one of them.
    """
    entry = row["entry"]
    out = []
    for metric in (resolve_metric(spec, entry), node_metric_of(spec, entry)):
        if metric is None:
            continue
        n, tot = _count_for(row, thr, metric)
        if tot:
            out.append((n, tot, "edges" if metric == "length" else "nodes"))
    return out


def flagged_count(row: dict, spec: dict, thr: dict) -> tuple[int, int]:
    """
    (flagged, total) for the block's own metric.

    FLAG_COUNT_SCOPE="crop" restricts both to what the panel actually shows, which
    is the honest number to print under a cropped map; "graph" reports the whole
    graph, which is the number to quote in prose.
    """
    return _count_for(row, thr, resolve_metric(spec, row["entry"]))


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


# Base gaps, in points. Not settings: they are the minimum that stops a letter
# touching the rule it sits against. PANEL_LETTER_PAD is the knob on top of them.
_LETTER_INSET = 2.0        # in from the panel edge, "inside"
_LETTER_LIFT  = 3.0        # above the panel edge, "above" -- the titles' own pad


def panel_letter(fig, dom, xd, yd, index: int):
    """
    One panel letter, in the corner PANEL_LETTER_LOC and PANEL_LETTER_SIDE name.

    LOC "above" puts it on the same baseline as the panel titles, outside the panel;
    "inside" drops it into the panel's own top corner, which always clears the title
    but sits over the map or the chart. SIDE picks which end of the panel, anchoring
    the letter so it never overhangs the edge it is measured from. PANEL_LETTER_PAD
    then shifts every letter in the figure by the same (dx, dy) points.

    Everything is computed in paper units rather than through plotly's xshift/yshift,
    so the wash rectangle -- which has no shift of its own -- lands on the text.
    """
    if not PANEL_LETTERS:
        return
    letter = chr(ord(PANEL_LETTER_START) + index)
    text = PANEL_LETTER_FMT.format(letter)
    if PANEL_LETTER_BOLD:
        text = f"<b>{text}</b>"
    size = PANEL_LETTER_SIZE or TITLE_FS
    if PANEL_LETTER_SIDE not in ("left", "right"):
        raise ValueError(f'PANEL_LETTER_SIDE must be "left" or "right", '
                         f'got {PANEL_LETTER_SIDE!r}')
    right = PANEL_LETTER_SIDE == "right"
    px2x = lambda v: v / dom["plot_w"]
    px2y = lambda v: v / dom["plot_h"]

    if PANEL_LETTER_LOC == "above":
        x, y, ya = (xd[1] if right else xd[0]), yd[1] + px2y(_LETTER_LIFT), "bottom"
    elif PANEL_LETTER_LOC == "inside":
        x = xd[1] - px2x(_LETTER_INSET) if right else xd[0] + px2x(_LETTER_INSET)
        y, ya = yd[1] - px2y(_LETTER_INSET), "top"
    else:
        raise ValueError(f'PANEL_LETTER_LOC must be "above" or "inside", got {PANEL_LETTER_LOC!r}')

    pad_x, pad_y = letter_pad(index)
    x, y = x + px2x(pad_x), y + px2y(pad_y)

    if PANEL_LETTER_LOC == "inside" and PANEL_LETTER_WASH:
        # ~0.62 em per character at this size, plus a little padding. The box grows
        # away from the anchored edge, so it follows SIDE the way the text does.
        w = px2x(len(PANEL_LETTER_FMT.format("a")) * size * 0.62 + 2.0)
        h = px2y(size * 1.25)
        x0, x1 = (x - w, x + px2x(1.0)) if right else (x - px2x(1.0), x + w)
        fig.add_shape(type="rect", xref="paper", yref="paper", layer="above",
                      x0=x0, x1=x1, y0=y - h, y1=y + px2y(1.0),
                      fillcolor="rgba(255,255,255,0.86)", line=dict(width=0))

    fig.add_annotation(xref="paper", yref="paper", x=x, y=y,
                       text=text, showarrow=False,
                       xanchor="right" if right else "left", yanchor=ya,
                       font=dict(size=size, color=TEXT, family=SANS))


def rounded_rect_path(x0, y0, x1, y1, rx, ry) -> str:
    """SVG path for a rounded rectangle, for plotly's shape `path`.

    Plotly's path grammar has no arc command, so each corner is a quadratic bezier
    whose control point IS the corner -- at these radii that is indistinguishable
    from a quarter circle, and it survives the PDF export as a curve rather than as
    a polygon.

    rx and ry are separate because the caller works in paper coordinates, where x
    and y are scaled by different pixel counts: passing one radius through the
    figure's own px->x and px->y conversions is what keeps the corner round on the
    page instead of stretched with the panel.
    """
    return (f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
            f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
            f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
            f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")


def axis_arrows(fig, dom, xd, yd):
    """Arrowhead at the far tip of each chart axis -- x to the right, y at the top.

    figure3_builder draws these as point-sized matplotlib markers; a plotly axis has
    no equivalent, so they are paper-space triangles built from the same point size
    and the plot's own pixel dimensions -- which is what keeps them matched to the
    spine whatever the panel's shape or the data limits. The apex sits exactly on
    the end of the spine, so the axis reads as tapering to a point rather than as a
    line with a mark parked beside it.
    """
    if not AXIS_ARROWS:
        return
    px2x, px2y = lambda p: p / dom["plot_w"], lambda p: p / dom["plot_h"]
    a = AXIS_ARROW_PX
    (x0, x1), (y0, y1) = xd, yd
    for tri in (
        # x axis: apex on the right end of the bottom spine
        ((x1, y0), (x1 - px2x(a), y0 + px2y(a / 2)), (x1 - px2x(a), y0 - px2y(a / 2))),
        # y axis: apex on the top end of the left spine
        ((x0, y1), (x0 - px2x(a / 2), y1 - px2y(a)), (x0 + px2x(a / 2), y1 - px2y(a))),
    ):
        fig.add_shape(type="path", xref="paper", yref="paper", layer="above",
                      path="M " + " L ".join(f"{u},{v}" for u, v in tri) + " Z",
                      fillcolor=AXIS_COLOR, line=dict(width=0))


def map_legend(fig, dom, xd, spec, row=None):
    """
    Two-line key inside a map, bottom-left.

    No bordered card: at 5 pt a box costs more attention than it returns, so the
    key sits on a plain white wash instead. The box is sized from the longest
    label rather than a fixed width, which is what kept clipping it.

    Under SLIM_LEGEND the key carries one row per claim and nothing else, each row
    naming the cut in the metric's own terms. `row` supplies the registry entry the
    node type is read from; without it the slim labels fall back to the generic ones.
    """
    loc = spec.get("legend_loc", LEGEND_LOC)
    if not SHOW_LEGENDS or loc in (None, "none"):
        return
    corners = ("bottom-left", "bottom-right", "top-left", "top-right")
    if loc not in corners:
        raise ValueError(f"legend_loc must be one of {corners} or 'none', got {loc!r}")
    s_    = LEGEND_SCALE
    fs    = LEGEND_FS * s_
    px2x  = lambda p: p / dom["plot_w"]
    px2y  = lambda p: p / dom["plot_h"]
    pad, row_h, gw = 3.0 * s_, LEGEND_FS * 1.3 * s_, 5.5 * s_

    is_length = spec["metric"] == "length"
    labels    = spec.get("legend_labels") or {}

    entry = row["entry"] if row else None

    def claim_glyph(metric_key):
        """
        The mark the map drew for what this row claims, so the key shows the same
        shape the reader is looking for: grid squares, mesh circles, undirected
        mesh edges, directed encoder/decoder edges. Without the registry entry the
        node side cannot be resolved, so a degree row falls back to the square.
        """
        if metric_key == "length":
            return edge_glyph(spec["subgraph"])
        if entry is None:
            return "square"
        resolved = resolve_metric(dict(metric=metric_key), entry)
        if resolved == "length":
            return edge_glyph(spec["subgraph"])
        return node_glyph(entry["src" if resolved == "src_deg" else "tgt"])

    glyph = claim_glyph(spec["metric"])

    def slim_text(metric_key, lo_key, hi_key):
        """The claim named in its own terms, or None if it cannot be resolved."""
        if entry is None:
            return None
        resolved = resolve_metric(dict(metric=metric_key), entry)
        node_type = (None if resolved == "length"
                     else entry["src" if resolved == "src_deg" else "tgt"])
        return threshold_phrase(resolved, node_type, spec["subgraph"],
                                spec.get(lo_key) if lo_key else None, spec.get(hi_key))

    def claim(lo_key, hi_key, kind, default_lo, default_hi, metric_key=None):
        low = spec.get(lo_key) is not None
        # An explicit legend_labels entry always wins: a block that phrased its own
        # row said something the generated phrase cannot know.
        text = labels.get(lo_key if low else hi_key)
        if text is None and SLIM_LEGEND:
            text = slim_text(metric_key, lo_key, hi_key)
        if text is None:
            text = default_lo if low else default_hi
        return ((VIOL_BELOW if low else VIOL_ABOVE), text, kind)

    def control(ctrl_key, kind, metric_key):
        """
        The alternative cut, named in full and in the lighter shade of the row above it.

        A band drawn in a colour the key never mentions is just a third state the
        reader has to guess at, so a control panel earns one extra row -- and the
        row has to state the control's own number, because "the other threshold" is
        exactly what the panel is asking to be compared.
        """
        text = labels.get(ctrl_key)
        if text is None and SLIM_LEGEND:
            phrase = slim_text(metric_key, None, ctrl_key)
            text = None if phrase is None else phrase + CONTROL_SUFFIX
        if text is None:
            text = "control threshold"
        return (control_color(), text, kind)

    # One row per claim. A dual-threshold block flags edges AND nodes, and each needs
    # its own glyph or the key says "over threshold" twice against different marks.
    items = [claim("lo", "hi", glyph, "under threshold", "over threshold",
                   spec["metric"])]
    if spec.get("hi_control") is not None:
        items.append(control("hi_control", glyph, spec["metric"]))
    if spec.get("node_metric"):
        node_glyph_ = claim_glyph(spec["node_metric"])
        items.append(claim("node_lo", "node_hi", node_glyph_,
                           "under threshold", "over threshold", spec["node_metric"]))
        if spec.get("node_hi_control") is not None:
            items.append(control("node_hi_control", node_glyph_, spec["node_metric"]))
    # Everything below is context, not claim. A slim key drops it: on a panel whose
    # whole point is the flagged marks, "within range" and the subgraph the degree was
    # counted over spend most of the key on what is NOT being asserted.
    if not SLIM_LEGEND:
        items.append((VIOL_NEUTRAL, labels.get("neutral", "within range"), glyph))
        # A degree panel's marks are nodes, but the degree was counted over a subgraph
        # and that subgraph is drawn underneath them. Name it, in its own edge colour,
        # or the reader has to guess what the isolated nodes are isolated from.
        if LEGEND_SUBGRAPH and not is_length:
            key = spec["subgraph"]
            items.append((context_edge_color(key, spec, LEGEND_EDGE_ALPHA),
                          f"{key} edges", edge_glyph(key)))

    # An arrow needs room for a shaft the eye reads as a shaft plus a head; a line,
    # square or circle does not. One width for every row, so the labels stay on a
    # single left edge whatever mix of marks the key ended up carrying.
    if any(k == "arrow" for _, _, k in items):
        gw = 8.0 * s_
    # ~0.5 em per character at this size; enough to keep the wash behind the text.
    text_w = max(len(l) for _, l, _ in items) * fs * 0.5
    box_w  = pad * 2 + gw + 2.5 * s_ + text_w
    box_h  = pad * 2 + row_h * len(items)

    # The key is placed against whichever corner the crop leaves empty. Paper y runs
    # 0..1 over the row, which is the map's full height, so "top" is 1 minus the box.
    vert, horiz = loc.split("-")
    inset = LEGEND_INSET_PX * s_
    x0 = xd[0] + px2x(inset) if horiz == "left" else xd[1] - px2x(box_w + inset)
    y0 = px2y(inset) if vert == "bottom" else 1.0 - px2y(box_h + inset)
    # Rounded, and outlined in LEGEND_BORDER. The radius is clamped to half the
    # shorter side so a one-row key cannot round itself into a lozenge.
    r = min(LEGEND_RADIUS_PX * s_, box_w / 2, box_h / 2)
    fig.add_shape(type="path", xref="paper", yref="paper", layer="above",
                  path=rounded_rect_path(x0, y0, x0 + px2x(box_w), y0 + px2y(box_h),
                                         px2x(r), px2y(r)),
                  fillcolor=LEGEND_BG,
                  line=dict(color=LEGEND_BORDER, width=LEGEND_BORDER_W * s_))

    gx = x0 + px2x(pad)
    for i, (color, label, kind) in enumerate(items):
        cy = y0 + px2y(box_h - pad - row_h * (i + 0.5))
        if kind == "line":
            fig.add_shape(type="line", xref="paper", yref="paper", layer="above",
                          x0=gx, x1=gx + px2x(gw), y0=cy, y1=cy,
                          line=dict(color=color, width=1.0 * s_))
        elif kind == "arrow":
            # Shaft plus a filled head. Filled rather than the chevron the app draws:
            # at LEGEND_FS an open head loses its two strokes to the rasteriser and
            # the row reads as a plain line again. The shaft stops short of the head
            # so the joint does not thicken into a blob.
            tip  = gx + px2x(gw)
            hx, hy = px2x(2.6 * s_), px2y(1.5 * s_)
            fig.add_shape(type="line", xref="paper", yref="paper", layer="above",
                          x0=gx, x1=tip - hx * 0.7, y0=cy, y1=cy,
                          line=dict(color=color, width=1.0 * s_))
            fig.add_shape(type="path", xref="paper", yref="paper", layer="above",
                          path=(f"M {tip - hx} {cy - hy} L {tip} {cy} "
                                f"L {tip - hx} {cy + hy} Z"),
                          fillcolor=color, line=dict(width=0))
        else:
            rx, ry, cx = px2x(2.2 * s_), px2y(2.2 * s_), gx + px2x(gw / 2)
            shape = "circle" if kind == "circle" else "rect"
            fig.add_shape(type=shape, xref="paper", yref="paper", layer="above",
                          x0=cx - rx, x1=cx + rx, y0=cy - ry, y1=cy + ry,
                          fillcolor=color, line=dict(width=0))
        fig.add_annotation(xref="paper", yref="paper", x=gx + px2x(gw + 2.5 * s_), y=cy,
                           text=label, showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=fs, color=LEGEND_FC, family=SANS))


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def split_domain(y, n: int, gap: float) -> list[tuple[float, float]]:
    """`n` stacked sub-domains inside `y`, top first, separated by `gap`."""
    lo, hi = y
    if n <= 1:
        return [(float(lo), float(hi))]
    h = (hi - lo - gap * (n - 1)) / n
    if h <= 0:
        raise ValueError(f"DIST_SPLIT_GAP={gap} leaves no room for {n} charts")
    clamp = lambda v: min(max(v, 0.0), 1.0)      # same float-noise guard as geometry()
    return [(clamp(hi - k * (h + gap) - h), clamp(hi - k * (h + gap))) for k in range(n)]


def chart_metrics(row: dict, spec: dict) -> list[str]:
    """The resolved metrics this block wants a distribution for, in caption order."""
    out = [resolve_metric(spec, row["entry"])]
    nm = node_metric_of(spec, row["entry"])
    if nm and DIST_SPLIT:
        out.append(nm)
    return out


def dist_panel(row: dict, spec: dict, thr: dict, axis, metric: str | None = None,
               panel_px: float | None = None):
    """
    The distribution beside a map: whichever chart `metric` calls for.

    `metric` defaults to the block's primary claim. A dual-claim block asks for each
    of its metrics in turn, which is what lets one column hold two charts.

    panel_px is the width the chart will occupy, needed to size the lollipop heads
    against their own spacing. None sizes them as if uncrowded.
    """
    panel_px = DIST_W * FIG_W_IN * PX_PER_IN if panel_px is None else panel_px
    metric = metric or resolve_metric(spec, row["entry"])
    if metric == "length":
        t = thr["length"]
        as_lollipop = (row["entry"]["plot"] == "lollipop" if LEN_CHART == "auto"
                       else LEN_CHART == "lollipop")
        vals = undirected_lengths(row)
        if as_lollipop:
            entries = cluster_by_gap(vals)
            head, x_range = lollipop_geometry(entries, t["full"], panel_px)
            traces = lollipop_traces(entries, t["lo"], t["hi"], t["active"], axis, head,
                                     t["hi_ctrl"])
        else:
            x_range = t["full"]
            traces = histogram_traces(vals, t["full"], t["lo"], t["hi"],
                                      t["active"], LEN_BINS, axis, t["hi_ctrl"])
        # Lengths are continuous, so a marker on the value cuts nothing: discrete=False.
        return traces, dict(title=title_case(f"{edge_type_label(row['key'])} edge length"),
                            x_range=x_range, thr=t, x_title="km", discrete=False)

    side = metric
    deg  = row["src_deg"] if side == "src_deg" else row["tgt_deg"]
    t    = thr[side]
    entries = value_counts(deg)
    if DEG_CHART_STYLE == "bar":
        # Bars are drawn at a fraction of the pitch already, and they sit ON the axis
        # start rather than overhanging it, so neither the sizing nor the pad applies.
        x_range = t["full"]
        traces  = bar_traces(entries, t["lo"], t["hi"], t["active"], axis, t["hi_ctrl"])
    else:
        head, x_range = lollipop_geometry(entries, t["full"], panel_px)
        traces = lollipop_traces(entries, t["lo"], t["hi"], t["active"], axis, head,
                                 t["hi_ctrl"])
    node_type = row["entry"]["src" if side == "src_deg" else "tgt"]
    return traces, dict(title=title_case(f"{node_type_label(node_type)} degree"), x_range=x_range,
                        thr=t, x_title="degree", discrete=True)


def context_edge_color(key: str, spec: dict, alpha: float) -> str:
    """
    The colour a block's non-thresholded edges draw in.

    Semantic on a length block. On a degree block the semantic colour is unusable:
    colors.json assigns the encoder and violation-above the same crimson, and the
    decoder and violation-below the same cyan, so m2g edges would come out the exact
    colour of the nodes flagged for lacking them. grey_of() is the tool's own
    de-emphasis (GNNExplorer.hexToGrayRgba), which is what "context, not claim" means
    in the UI, so the panel keeps one unambiguous colour per role.
    """
    color = {"g2m": EDGE_ENCODER, "m2g": EDGE_DECODER}.get(key, EDGE_MESH)
    grey  = (DEGREE_BLOCK_EDGE_GREY and spec["metric"] != "length"
             and not spec.get("show_subgraph", SUBGRAPH_VIEW))
    return grey_of(color, alpha) if grey else rgba(color, alpha)


def map_traces(row: dict, spec: dict, thr: dict, geo: str, panel_w_px: float) -> list:
    """
    Inspect panel for one block.

    Same encodings as `inspect_traces` in figure2_builder, but only the block's
    own metric recolours anything: a length block leaves its nodes in their
    semantic colour, a degree block leaves its edges in theirs. That keeps one
    claim per panel.
    """
    entry, n_lv = row["entry"], row["n_levels"]
    size   = base_node_px(row, panel_w_px)
    border = max(NODE_BORDER_MIN_PX, size * NODE_BORDER_FRAC)
    payload, key = row["inspect_edges"], row["key"]
    lengths = payload["length"]
    full_view = spec.get("show_subgraph", SUBGRAPH_VIEW) and spec["metric"] != "length"
    node_m = node_metric_of(spec, entry)
    out = ne_basemap_traces(row, geo) if BASEMAP_SOURCE == "naturalearth" else []

    if key in ("g2m", "m2g"):
        base_w = 1.8
    elif key.startswith("m2m"):
        base_w = stroke_for_level(int(key.split("_")[1]) if "_" in key else 0, n_lv)
    else:
        base_w = stroke_for_level(int(key.split("_")[2]), n_lv)
    base_w *= EDGE_WIDTH_SCALE

    t = thr["length"]
    if spec["metric"] == "length" and t["active"]:
        below = lengths <= t["lo"]
        # A control cut splits the flagged side in two: the edges both cuts agree on,
        # in the full colour, and the ones only one of them catches, in the lighter
        # shade. Drawn between the neutral edges and the flagged ones, so the marks
        # the claim actually rests on stay on top.
        band  = control_band(t["hi"], t["hi_ctrl"])
        ctrl  = ((lengths >= band[0]) & (lengths < band[1]) if band is not None
                 else np.zeros(len(lengths), bool))
        above = lengths >= (band[1] if band is not None else t["hi"])
        mid = ~(below | above | ctrl)
        out.append(edge_trace(geo, payload, mid,
                              rgba(VIOL_NEUTRAL, EDGE_ALPHA_VIOL_NEUTRAL), base_w * 0.55))
        out.append(edge_trace(geo, payload, ctrl,
                              rgba(control_color(), EDGE_ALPHA_VIOL_FLAG), base_w))
        out.append(edge_trace(geo, payload, below, rgba(VIOL_BELOW, EDGE_ALPHA_VIOL_FLAG), base_w))
        out.append(edge_trace(geo, payload, above, rgba(VIOL_ABOVE, EDGE_ALPHA_VIOL_FLAG), base_w))
    else:
        # Not the block's metric, so the edges stay in their semantic colour. On a
        # degree panel they are not decoration: they ARE the subgraph the degree was
        # counted over, and an isolated node only reads as isolated against them.
        # They still draw below EDGE_ALPHA_NORMAL, because at 0.25 deg a fully opaque
        # g2m fuses into bands that bury the flagged nodes.
        default_a = (EDGE_ALPHA_NORMAL if spec["metric"] == "length"
                     else SUBGRAPH_EDGE_ALPHA if full_view
                     else DEGREE_BLOCK_EDGE_ALPHA)
        alpha = default_a if spec.get("edge_alpha") is None else spec["edge_alpha"]
        if alpha > 0:
            color = context_edge_color(key, spec, alpha)
            out.append(edge_trace(geo, payload, np.ones(len(lengths), bool), color, base_w))
            # Direction is the whole difference between deg_out and deg_in, so a full
            # view of a directed subgraph shows it rather than asserting it in the title.
            n_edges = int(len(lengths))
            if (full_view and SUBGRAPH_ARROWS and key in ("g2m", "m2g")
                    and n_edges <= SUBGRAPH_ARROW_MAX_E):
                tgt_px = (mesh_node_px(row, size, entry["tgt"], True, True)
                          if entry["tgt"] != "grid" else size)
                out.append(arrow_trace(geo, payload, np.ones(len(lengths), bool),
                                       color, tgt_px * SUBGRAPH_ARROW_FRAC))

    shown = [entry["src"]] if entry["src"] == entry["tgt"] else [entry["src"], entry["tgt"]]
    # The thresholded node type draws LAST so nothing lands on top of the flagged
    # marks -- in full view the mesh endpoints are large enough to cover them.
    _last = node_m or spec["metric"]
    shown.sort(key=lambda t: (t == entry["src"]) == (_last == "src_deg"))
    for t_name in shown:
        ll, m = row["nodes"][t_name], row["masks"][t_name]
        is_src = (t_name == entry["src"])
        side   = "src_deg" if is_src else "tgt_deg"
        deg    = row["src_deg"] if is_src else row["tgt_deg"]
        td     = thr[side]
        symbol = "square" if t_name == "grid" else "circle"
        boost  = 1.0 if t_name == "grid" else node_level_boost(t_name, n_lv)
        # "endpoint" = a side this block is not thresholding at all.
        thresholded = {spec["metric"], node_m} - {None, "length"}
        endpoint = bool(thresholded) and side not in thresholded
        s      = (size if t_name == "grid"
                  else mesh_node_px(row, size, t_name, full_view, endpoint))
        b      = border if t_name == "grid" else border * boost

        if side not in thresholded or not td["active"]:
            base = NODE_GRID if t_name == "grid" else NODE_MESH
            out.append(node_trace(geo, ll, m, symbol, s, rgba(base, NODE_ICON_ALPHA), b))
            continue
        # A flagged node draws larger as well as darker. On the 0.25 deg ocean grid a
        # neutral node is under a point across, so without the boost the defect is
        # indistinguishable from the field it sits in.
        d = np.asarray(deg)
        # Same three-way split the edges get: agreed, disputed, neutral. hi_full is the
        # outer of the two cuts, so the disputed degrees are not counted as flagged
        # whichever side of hi the control was set.
        nband   = control_band(td["hi"], td["hi_ctrl"])
        hi_full = nband[1] if nband is not None else td["hi"]
        in_band = (((d >= nband[0]) & (d < nband[1])) if nband is not None
                   else np.zeros(d.shape, bool))
        for gmask, color, boost_s in (
            (m & (d > td["lo"]) & (d < hi_full) & ~in_band,
                                                  rgba(VIOL_NEUTRAL, NODE_VIOL_NEUTRAL_ALPHA), 1.0),
            (m & in_band,                         rgba(control_color(), NODE_VIOL_ALPHA), VIOL_NODE_BOOST),
            (m & (d <= td["lo"]),                 rgba(VIOL_BELOW,   NODE_VIOL_ALPHA), VIOL_NODE_BOOST),
            (m & (d >= hi_full),                  rgba(VIOL_ABOVE,   NODE_VIOL_ALPHA), VIOL_NODE_BOOST),
        ):
            out.append(node_trace(geo, ll, gmask, symbol, s * boost_s, color, b))
    return [t for t in out if t is not None]


def build_figure(blocks=None):
    blocks = blocks or DEFAULT_BLOCKS
    if len(blocks) != 2:
        raise ValueError("this layout is two blocks wide")
    dom = geometry()
    fig = go.Figure()
    layout = dict(width=int(round(dom["fig_w"])), height=int(round(dom["fig_h"])),
                  margin=dict(l=dom["margin"]["l"], r=dom["margin"]["r"],
                              t=dom["margin"]["t"], b=dom["margin"]["b"]),
                  paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
                  font=dict(family=SANS, size=TICKNUM_FS, color=TICKNUM_FC))

    axis_n = 0          # runs across blocks: a split column claims two axis pairs
    panel_n = 0         # panel letters, in reading order: left to right, and top to
                        #   bottom within a split distribution column
    for i, spec in enumerate(blocks):
      # Everything this block draws happens under its own settings=, if it has any.
      # dom is already fixed above, so a block cannot disturb the shared layout.
      with overrides(**block_settings(spec)):
          row = build_row(dict(graph=spec["graph"], subgraph=spec["subgraph"],
                               label=spec.get("title"), crop=block_crop(spec, dom)))
          # Resolve "grid_deg"/"mesh_deg" to the concrete side once, here, so every
          # panel-drawing function below sees a plain "src_deg"/"tgt_deg"/"length".
          spec = dict(spec, metric=resolve_metric(spec, row["entry"]))
          thr = block_thresholds(row, spec)

          # map
          geo_id  = "geo" if i == 0 else f"geo{i+1}"
          geo_ref = geo_id
          layout[geo_id] = geo_layout(row) | dict(
              domain=dict(x=list(dom["x"][2 * i]), y=list(dom["y"])))
          panel_letter(fig, dom, dom["x"][2 * i], dom["y"], panel_n)
          panel_n += 1
          for tr in map_traces(row, spec, thr, geo_ref, dom["map_px"]):
              fig.add_trace(tr)

          # distribution(s) -- one per claim, stacked in the block's own column
          xd     = dom["x"][2 * i + 1]
          metrics = chart_metrics(row, spec)
          slots   = split_domain(dom["y"], len(metrics), DIST_SPLIT_GAP)
          infos   = []
          axis_common = dict(showgrid=False, zeroline=False, showline=True,
                             linecolor=AXIS_COLOR, linewidth=AXIS_WIDTH, ticks="outside",
                             ticklen=AXIS_TICK_LEN, tickwidth=AXIS_WIDTH,
                             tickcolor=AXIS_COLOR,
                             tickfont=dict(size=TICKNUM_FS, color=TICKNUM_FC, family=SANS))
          for cm, yd in zip(metrics, slots):
              axis_n += 1
              xa = "x" if axis_n == 1 else f"x{axis_n}"
              ya = "y" if axis_n == 1 else f"y{axis_n}"
              traces, info = dist_panel(row, spec, thr, (xa, ya), metric=cm,
                                        panel_px=dom["dist_px"])
              infos.append((info, yd))
              for tr in traces:
                  fig.add_trace(tr)
              ymax = max((max(t.y) for t in traces if getattr(t, "y", None) is not None
                          and len(t.y) and np.isfinite(np.nanmax(t.y))), default=1.0)
              suffix = "" if axis_n == 1 else str(axis_n)
              layout["xaxis" + suffix] = dict(
                  domain=list(xd), anchor=ya, range=list(info["x_range"]), nticks=4,
                  title=dict(text=info["x_title"],
                             font=dict(size=AXLAB_FS, color=AXLAB_FC, family=SANS), standoff=1),
                  **axis_common)
              yax = dict(domain=list(yd), anchor=xa, nticks=3, **axis_common)
              if SHOW_Y_TITLE:
                  yax["title"] = dict(text="count",
                                      font=dict(size=AXLAB_FS, color=AXLAB_FC, family=SANS),
                                      standoff=1)
              # A lettered chart needs room for the letter above its tallest mark.
              head = (PANEL_LETTER_HEADROOM
                      if (PANEL_LETTERS and PANEL_LETTER_SCOPE == "all"
                          and PANEL_LETTER_LOC == "inside") else 1.0)
              if CHART_LOG_Y:
                  yax |= dict(type="log", range=[0, math.log10(max(ymax, 1)) * 1.06 * head])
              else:
                  yax |= dict(range=[0, ymax * 1.08 * head])
              layout["yaxis" + suffix] = yax
              axis_arrows(fig, dom, xd, yd)
              for sh in threshold_shapes(xa, ya, info):
                  fig.add_shape(**sh)
              if PANEL_LETTER_SCOPE == "all":
                  panel_letter(fig, dom, xd, yd, panel_n)
                  panel_n += 1

          # titles. add_annotation throughout -- update_layout(annotations=[...]) merges
          # by index, so setting the list at the end would silently clobber whatever the
          # legend had already added.
          parts = flagged_parts(row, spec, thr)
          title = spec.get("title") or f"{spec['graph']} · {spec['subgraph']}"
          fig.add_annotation(xref="paper", yref="paper",
                             x=(dom["x"][2*i][0] + dom["x"][2*i][1]) / 2, y=1.0, yshift=3,
                             text=title, showarrow=False, xanchor="center", yanchor="bottom",
                             font=dict(size=TITLE_FS, color=TITLE_FC, family=SANS))
          if DIST_TITLE:
              # Each chart is titled over its own slot, so a split column does not leave
              # the lower half anonymous. Only the top one clears the panel; the rest sit
              # in the gap above themselves.
              for info, yd in infos:
                  fig.add_annotation(xref="paper", yref="paper",
                                     x=(xd[0] + xd[1]) / 2, y=yd[1], yshift=3,
                                     text=info["title"], showarrow=False, xanchor="center",
                                     yanchor="bottom",
                                     font=dict(size=TITLE_FS, color=TITLE_FC, family=SANS))
          if any(n for n, _, _ in parts):
              fig.add_annotation(xref="paper", yref="paper",
                                 x=dom["x"][2*i][0], y=0.0, yshift=-2, xshift=1,
                                 text=" · ".join(f"{n:,} of {t:,} {u}" for n, t, u in parts)
                                      + " flagged",
                                 showarrow=False, xanchor="left", yanchor="top",
                                 font=dict(size=LEGEND_FS, color=LEGEND_FC, family=SANS))
          if SHOW_LEGENDS:
              map_legend(fig, dom, dom["x"][2 * i], spec, row)

    check_letter_pad(panel_n)

    if DIVIDER:
        mid, half = 0.5, 0.5 * DIV_SCALE
        fig.add_shape(type="line", xref="paper", yref="paper",
                      x0=dom["x_div"], x1=dom["x_div"],
                      y0=mid - half, y1=mid + half, layer="above", opacity=DIV_ALPHA,
                      line=dict(color=TEXT_MUTED, width=DIV_WIDTH, dash=DIV_DASH))

    fig.update_layout(**layout)
    return fig


def build(blocks=None, **settings):
    """
    Build the figure.

    blocks    the two panel specs. Defaults to DEFAULT_BLOCKS.
    settings  any UPPERCASE name in config_keys(), applied for this call only.
              show_config(common=True) lists the ones worth reaching for;
              an unknown name raises instead of being ignored.

    A block spec is:
        graph      folder under GRAPH_SEARCH_DIRS, or an absolute path
        subgraph   registry key: "m2m" | "m2m_0".. | "g2m" | "m2g" | "mesh_up_0"..
        metric     "length"                  threshold edges, chart = histogram
                   "grid_deg" | "mesh_deg"   threshold that node type's degree in
                                             this subgraph -- resolves to the source
                                             side for g2m, the target side for m2g
                   "src_deg" | "tgt_deg"     the explicit sides, still accepted
        lo, hi     threshold in the metric's units; None = that side inactive
        hi_control a second cut on the same metric, beside hi: the control case. The
                   band between the two cuts -- the marks one flags and the other
                   does not -- draws in a lighter shade of the flag colour in both
                   the chart and the map, with its own key row, so how much rests on
                   where the line was put is visible instead of asserted. Requires
                   hi; a value under hi is the looser alternative, over hi the
                   stricter one. node_hi_control does the same for node_metric.
        region     key into REGIONS, or None if `crop` is given
        crop       dict(center=(lat, lon), height=<deg>), overrides `region`
        zoom       crop magnification for this block; multiplies the global ZOOM
        shift      (dx, dy) sliding the crop off its centre, in crop widths and
                   crop heights: +dx right, -dx left, +dy up, -dy down. Size and
                   square cells are untouched -- only the window moves. Applied
                   after zoom, so (0.5, 0) is half the drawn panel at any zoom.
        edge_alpha 0-255 override for this block's edges; 0 hides them
        show_subgraph  draw the whole subgraph the degree was counted over --
                   semantic edge colour, endpoints at full size, arrowheads on a
                   directed subgraph -- instead of the de-emphasised default.
                   Defaults to the SUBGRAPH_VIEW setting. Ignored on a length block.
        node_metric  a SECOND claim on the same panel: threshold nodes on this
                   degree metric ("grid_deg" | "mesh_deg" | "src_deg" | "tgt_deg")
                   while `metric` thresholds edges. With node_lo / node_hi. Use it
                   when two defects are the same story in one place -- long edges
                   over Denmark and the Baltic nodes they strand -- and splitting
                   them across panels would destroy the adjacency that is the point.
        node_lo, node_hi  thresholds for node_metric, in degrees
        legend_labels  {"lo"|"hi"|"hi_control"|"node_lo"|"node_hi"|
                   "node_hi_control"|"neutral": text} to rename
                   the key's rows, which generic "over threshold" cannot carry
                   when a panel makes two different claims
        settings   {"SETTING": value} applied to THIS block only -- its marks, its
                   charts, its key, its crop. Use it when the two panels need
                   different treatment: a dense crop wanting a larger
                   VIOL_NODE_BOOST, or one map on a different BASEMAP_SOURCE.
                   Layout settings (FIG_W_IN, MAP_W, GAP_*, MARGIN_IN, DIVIDER, ...)
                   are read once for the whole figure and are rejected here.
        legend_loc which corner the in-map key sits in: "bottom-left" (default),
                   "bottom-right", "top-left", "top-right", or "none" to drop it
                   from this panel only. Defaults to the LEGEND_LOC setting.
        title      panel title; None -> auto from graph + subgraph
    """
    with overrides(**settings):
        return build_figure(blocks if blocks is not None else DEFAULT_BLOCKS)


def with_blocks(blocks, stem=None, formats=("pdf", "png"), dpi: int = 400,
                out_dir=None, save_tight: bool = None,
                save_transparent: bool = None, **settings):
    """
    Render an alternate pair of blocks, optionally saving it.

    blocks    the two panel specs
    stem      filename stem; None just returns the figure without writing
    formats   which files to write -- the same set export() takes, and written the
              same way, so a variant is saved at true size and true DPI rather than
              the scale-1 PDF this used to emit
    dpi       PNG resolution; the PDF is always vector at FIG_W_IN inches
    out_dir   defaults to the notebooks directory
    save_tight override SAVE_TIGHT for this call -- crop the written files to the ink
    save_transparent override SAVE_TRANSPARENT -- drop the white background
    settings  figure-wide settings, exactly as build() takes them. For settings that
              differ BETWEEN the two panels, put them in the block's own settings=.
    """
    fig = build(blocks, **settings)
    if stem:
        export(fig, stem, formats=formats, dpi=dpi, out_dir=out_dir,
               save_tight=save_tight, save_transparent=save_transparent)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SHOWING AND EXPORTING
# ═══════════════════════════════════════════════════════════════════════════

def use_png_renderer(scale: int = 3):
    """
    Make fig.show() legible in a notebook.

    The figure's width is set in POINTS so the PDF lands at column width -- 5.5 in
    is 396 px -- and the default renderer draws that as 396 CSS px, a postage stamp.
    Plotly's png renderer re-renders at `scale` x and the notebook shows the result
    at its natural pixel size. Same figure, bigger on screen. Pass
    pio.renderers.default = "plotly_mimetype" to get the interactive (tiny) one back.
    """
    pio.renderers.default = "png"
    pio.renderers["png"].scale = scale


def preview(fig, dpi: int = 240, width: str = "100%"):
    """
    Show `fig` scaled to the cell width, whatever the renderer is set to.

    use_png_renderer() is enough on its own; this is for when you want the figure
    to fill the cell exactly, or a different resolution for one figure.
    """
    from IPython.display import HTML, display
    png = fig.to_image(format="png", scale=dpi / PX_PER_IN)
    b64 = base64.b64encode(png).decode()
    display(HTML(f'<img src="data:image/png;base64,{b64}" alt="figure 2" '
                 f'style="width:{width};max-width:100%;height:auto;display:block">'))
    return fig


# Fully transparent. plotly still emits the paper and plot-area rectangles rather
# than omitting them -- they come out as fills under an ExtGState with ca=0, which
# renders to nothing and rasterises to alpha 0. Written white-under-zero rather
# than black-under-zero on the principle that the colour a flattener might fall
# back to should be the one the figure used to have.
NO_PAPER = "rgba(255,255,255,0)"


def _paper_rgba(fig) -> tuple:
    """The background the crop measures against, as (r, g, b, a) with a in 0-255."""
    c = fig.layout.paper_bgcolor or "white"
    if c in ("white", "#fff", "#ffffff"):
        return (255, 255, 255, 255)
    if c.startswith("#"):
        h = c.lstrip("#")
        h = "".join(ch * 2 for ch in h) if len(h) == 3 else h
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    if c.startswith("rgb"):
        v = [float(x) for x in c[c.index("(") + 1:c.index(")")].split(",")]
        return tuple(int(x) for x in v[:3]) + (int(round(v[3] * 255)) if len(v) > 3
                                               else 255,)
    raise ValueError(f"cannot read paper_bgcolor {c!r} -- set SAVE_TIGHT=False")


def content_bbox(fig, png=None, tol: int = 2):
    """
    Where the figure actually draws, as (x0, y0, x1, y1) in NOMINAL px (= points)
    measured from the top-left of the paper. None if nothing but background is there.

    Measured on a raster rather than reasoned about: the content is annotations,
    shapes, tick labels and a basemap, and only the renderer knows where a 4.6 pt
    label ends up. `png` reuses a file already written at export DPI -- the finer
    the raster, the finer the edge -- otherwise a throwaway render at
    SAVE_PROBE_SCALE is made here.

    On a transparent paper the test is opacity: any pixel the figure touched at
    all is content, which trims to the ink itself. On an opaque one it can only be
    "differs from the paper colour", which stops at the edge of the paint.
    """
    from PIL import Image
    import io

    if fig.layout.width is None:
        raise ValueError("the figure has no explicit layout.width, so a raster "
                         "cannot be mapped back to points -- set SAVE_TIGHT=False")
    im = (Image.open(png) if png else Image.open(io.BytesIO(
        fig.to_image(format="png", scale=SAVE_PROBE_SCALE)))).convert("RGBA")
    a = np.asarray(im).astype(np.int16)
    r, g, b, alpha = _paper_rgba(fig)
    if alpha == 0:
        bg = a[..., 3] <= tol
    else:
        bg = ((np.abs(a[..., 0] - r) <= tol) & (np.abs(a[..., 1] - g) <= tol)
              & (np.abs(a[..., 2] - b) <= tol) & (a[..., 3] >= 255 - tol))
    ys, xs = np.nonzero(~bg)
    if not len(xs):
        return None
    # The raster is only an integer number of px wide, so recover its true px per
    # nominal px from the file rather than assuming the scale it was asked for.
    s = im.width / float(fig.layout.width)
    return (xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s)


def _crop_box(fig, bbox, pad_in):
    """bbox + SAVE_PAD, clamped to the paper, in nominal px from the top-left."""
    pad = pad_in * PX_PER_IN
    w, h = float(fig.layout.width), float(fig.layout.height)
    x0, y0, x1, y1 = bbox
    return (max(0.0, x0 - pad), max(0.0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def _crop_raster(path, box, fig_w):
    """Crop a written PNG/JPEG/WebP in place to `box`, given in nominal px."""
    from PIL import Image

    im = Image.open(path)
    s = im.width / fig_w                      # file px per nominal px
    l, t, r, b = (int(round(v * s)) for v in box)
    im.crop((max(0, l), max(0, t), min(im.width, r), min(im.height, b))).save(path)


_MEDIABOX = _re.compile(rb"/MediaBox\s*\[([^\]]*)\]")

# PDF points per nominal px. Chrome lays out at 96 CSS px to the inch and PDF is 72
# pt to the inch, so a 396 px figure prints onto a 297 pt page.
PDF_PT_PER_PX = 72 / 96


def _crop_pdf(path, box, fig_w, fig_h, bleed=None):
    """
    Crop a written PDF in place by narrowing its MediaBox to `box`.

    The page content is left exactly as kaleido drew it -- only the window onto it
    moves, which is what a cropping tool does and why the vector output survives.
    The new box is padded back to the old one's byte length so every xref offset
    in the file stays valid; if it will not fit, the file is left uncropped and
    that is reported rather than a broken PDF written.
    """
    data = path.read_bytes()
    m = _MEDIABOX.search(data)
    if not m:
        print(f"  note: no MediaBox in {path.name} (compressed object streams?) "
              f"-- left uncropped")
        return False
    mx0, my0, mx1, my1 = (float(v) for v in m.group(1).split())
    bleed = SAVE_PDF_BLEED if bleed is None else bleed
    # Chrome lays the page out in CSS px and writes it in points, so a nominal px
    # is PDF_PT_PER_PX of a point and the figure's top-left sits on the page's --
    # exact, once full_page_pdf() has removed the print margin. Deriving the scale
    # from the MediaBox instead would inherit Chrome's rounding of the page size,
    # which is a fifth of a point out over this figure's height.
    sx = sy = PDF_PT_PER_PX
    x0, y0, x1, y1 = box
    # PDF user space runs up from the bottom-left; the box runs down from the top.
    # Chrome's page is rounded to its own precision and can fall a fraction short
    # of the figure, so keep the new box inside the old one.
    v = (max(mx0, mx0 + x0 * sx - bleed), max(my0, my1 - y1 * sy - bleed),
         min(mx1, mx0 + x1 * sx + bleed), min(my1, my1 - y0 * sy + bleed))
    # Spend whatever decimals the old box's byte budget affords. Even 0 of them is
    # a 1/72 in error on a crop whose whole point is to remove tenths of an inch.
    budget = len(m.group(0))
    for dp in (4, 3, 2, 1, 0):
        new = ("/MediaBox[" + " ".join(f"{c:.{dp}f}" for c in v) + "]").encode()
        if len(new) <= budget:
            break
    else:
        print(f"  note: cropped MediaBox does not fit {path.name}'s byte budget "
              f"-- left uncropped")
        return False
    new += b" " * (budget - len(new))            # whitespace between dict entries
    path.write_bytes(data[:m.start()] + new + data[m.end():])
    return True


def unpapered(fig):
    """
    A copy of `fig` with the white sheet behind it removed.

    Only the paper and the empty chart backgrounds are cleared. Everything the
    figure means by white it says explicitly -- the geo subplots paint their own
    ocean, the legend and panel-letter washes their own 86%% white -- so those are
    untouched and the maps look exactly as they did.
    """
    return go.Figure(fig).update_layout(paper_bgcolor=NO_PAPER, plot_bgcolor=NO_PAPER)


@contextmanager
def full_page_pdf():
    """
    Print PDFs with no page margin for the duration of the block. Yields whether
    it managed to, which is also whether the PDF can be trusted to be a 1:1 window
    on the figure.

    kaleido 1.x does not draw a PDF, it asks Chrome to PRINT one, and it hardcodes
    a 0.1 in margin on all four sides plus pageRanges="1". Chrome scales the figure
    down to fit the printable WIDTH and then discards whatever no longer fits the
    printable HEIGHT -- and for a figure wider than it is tall that is guaranteed to
    happen. figure2 at 5.5 x 1.54 in came out scaled by 0.955 and cut about 0.18 in
    short: the captions, the x tick labels and the axis titles, gone from the PDF
    only, with no error. The PNG comes down a different path and is unaffected,
    which is why the two disagreed.

    With the margins at zero the page is the figure again -- 0.75 pt per nominal px,
    no offset, no clip -- which is also the assumption the MediaBox crop relies on.
    """
    try:
        from kaleido._kaleido_tab import _devtools_utils as dt
        original = dt.print_pdf
        raise_error = dt._raise_error
    except (ImportError, AttributeError):
        print("  note: this kaleido does not expose its PDF print options -- the PDF "
              "keeps Chrome's 0.1 in page margin, which SCALES AND CLIPS a figure "
              "wider than it is tall. Export PNG and check the PDF by eye.")
        yield False
        return

    async def _print_pdf(tab):
        params = {"printBackground": True, "preferCSSPageSize": True,
                  "pageRanges": "1", "marginTop": 0, "marginBottom": 0,
                  "marginLeft": 0, "marginRight": 0}
        response = await tab.send_command("Page.printToPDF", params=params)
        raise_error(response)
        return response.get("result", {}).get("data")

    dt.print_pdf = _print_pdf
    try:
        yield True
    finally:
        dt.print_pdf = original


def export(fig, stem="figure2_ocean_defects", formats=("pdf", "png"), dpi: int = 400,
           out_dir=None, save_tight: bool = None, save_transparent: bool = None):
    """
    Write the figure at its true size.

    Nominal px are PDF points (PX_PER_IN = 72), so the figure is FIG_W_IN inches
    wide and the PNG scale is the DPI over 72.

    With SAVE_TRANSPARENT the white background is dropped first, so the file
    carries no sheet onto the page. With SAVE_TIGHT the written FILES are then
    cropped to the ink -- the raster trimmed, the PDF's MediaBox narrowed -- which
    removes the dead band under the tick labels. Pass save_transparent= /
    save_tight= to override either for one call.

    Nothing is re-rendered from different geometry and nothing moves: the panels,
    type sizes and rules are the ones plotly already drew.
    """
    tight = SAVE_TIGHT if save_tight is None else save_tight
    if SAVE_TRANSPARENT if save_transparent is None else save_transparent:
        fig = unpapered(fig)
    out_dir = Path(out_dir) if out_dir else ROOT / "notebooks"
    paths = []
    with full_page_pdf() as full_page:
        for fmt in formats:
            path = (out_dir / stem).with_suffix(f".{fmt}")
            fig.write_image(path, scale=(dpi / PX_PER_IN) if fmt == "png" else 1)
            print("wrote", path)
            paths.append(path)

    w, h = float(fig.layout.width), float(fig.layout.height)
    box = None
    if tight:
        # Measure on the PNG if one was just written -- it is the finest raster
        # available and costs nothing -- otherwise render a probe.
        png = next((p for p in paths if p.suffix == ".png"), None)
        bbox = content_bbox(fig, png)
        if bbox is None:
            print("  note: nothing but background found -- left uncropped")
        else:
            # Ink ON the paper edge means plotly clipped something before the file
            # was ever written -- a title wider than its panel plus margin, say.
            # No crop can recover it (unlike matplotlib, plotly rasterises the cut),
            # so say which side and let the margin be widened.
            clipped = [n for n, hit in (("left", bbox[0] <= 0.5), ("top", bbox[1] <= 0.5),
                                        ("right", bbox[2] >= w - 0.5),
                                        ("bottom", bbox[3] >= h - 0.5)) if hit]
            if clipped:
                print(f"  note: ink reaches the {' and '.join(clipped)} paper edge -- "
                      f"content was CLIPPED at render time. Widen MARGIN_IN "
                      f"{'/'.join(sorted({'left':'l','right':'r','top':'t','bottom':'b'}[c] for c in clipped))}"
                      f" and re-run; the crop cannot bring back what was cut.")
            box = _crop_box(fig, bbox, SAVE_PAD)
            for path in paths:
                if path.suffix in (".png", ".jpg", ".jpeg", ".webp"):
                    _crop_raster(path, box, w)
                elif path.suffix == ".pdf":
                    # Only safe on a full page: otherwise the MediaBox is not the
                    # figure's own box and cropping in figure coordinates would
                    # trim the wrong region.
                    if full_page:
                        _crop_pdf(path, box, w, h)
                    else:
                        print(f"  note: {path.name} left uncropped -- its page is "
                              f"not a 1:1 window on the figure")
                else:
                    print(f"  note: {path.suffix} cannot be cropped -- left as drawn")

    if box is None:
        print(f"  {w / PX_PER_IN:.3f} x {h / PX_PER_IN:.3f} in")
    else:
        print(f"  {(box[2] - box[0]) / PX_PER_IN:.3f} x "
              f"{(box[3] - box[1]) / PX_PER_IN:.3f} in "
              f"(cropped from {w / PX_PER_IN:.3f} x {h / PX_PER_IN:.3f})")
    return paths
