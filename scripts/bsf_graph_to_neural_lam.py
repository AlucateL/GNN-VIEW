"""
CLI wrapper for the ocean graph converter.

The conversion logic lives in `backend/src/services/bsf_converter.py` so the backend can
call it at upload time; this script is the offline entry point for converting a graph
directory by hand (e.g. to ship one as a preset under backend/data/graphs/).

Usage:
    python scripts/bsf_graph_to_neural_lam.py <graph_dir> <output_dir> [--lcc ...]

Examples:
    # Global graph -- self-contained, nothing external needed
    python scripts/bsf_graph_to_neural_lam.py \\
        backend/data/graphs/global_cluster_analysis /tmp/out

    # Limited-area graph -- needs the projection its coordinates were built in
    python scripts/bsf_graph_to_neural_lam.py \\
        backend/data/graphs/baltic_cluster /tmp/out --lcc 60,20,60,60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from services.bsf_converter import convert_bsf_to_neural_lam  # noqa: E402


def cli(input_args=None):
    parser = argparse.ArgumentParser(
        description="Convert an ocean graph directory into a neural-lam upload bundle."
    )
    parser.add_argument("bsf_graph_dir", type=Path, help="graph directory to convert")
    parser.add_argument("output_dir", type=Path, help="where to write the neural-lam .pt set")
    parser.add_argument(
        "--lcc",
        default=None,
        metavar="lat_0,lon_0,lat_1,lat_2",
        help="Lambert Conformal Conic parameters of the graph's projected coordinates, "
        "e.g. '60,20,60,60'. Required for limited-area graphs, ignored for global ones. "
        "Pass only lat_0,lon_0 for the tangent case (lat_1 = lat_2 = lat_0).",
    )
    args = parser.parse_args(input_args)

    projection = None
    if args.lcc:
        parts = [float(v) for v in args.lcc.split(",")]
        if len(parts) == 2:
            parts += [parts[0], parts[0]]
        if len(parts) != 4:
            parser.error("--lcc takes 2 or 4 comma-separated numbers")
        projection = dict(zip(("lat_0", "lon_0", "lat_1", "lat_2"), parts))

    summary = convert_bsf_to_neural_lam(args.bsf_graph_dir, args.output_dir, projection)
    print(json.dumps(summary, indent=2))
    print("\nUpload the contents of output_dir via POST /graphs, or copy it into "
          "backend/data/graphs/<name>/ to ship it as a preset.")


if __name__ == "__main__":
    cli()