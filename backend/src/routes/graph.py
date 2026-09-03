from fastapi import APIRouter, HTTPException
from pathlib import Path

import json
import struct

import orjson
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from sqlalchemy.orm import Session
from services.database import get_db
from services.crud import GraphCRUD
from services.file_utils import FileHandler
from services.json_response import ORJSONResponse
from models.schemas import GraphMeta, GraphCreate, GraphUpdate, GraphUpload
from pathlib import Path

from services.graph import (
    get_graph_metrics,
    _two_std_bounds,
    edges_to_list,
    list_graph_folders,
    load_graph,
    nodes_to_list,
)
from services.anemoi_converter import is_anemoi_format, convert_anemoi_to_neural_lam
from services.bsf_converter import is_bsf_format, convert_bsf_to_neural_lam
## Bbox with rtree/torch — swap the import to switch implementation
# ── torch (default) ─────────────────────────────────────────────────
from services.bbox_torch import bbox_node_mask, filter_edges_by_node_masks
#  ── rtree ────────────────────────────────────────────────────────
# from services.bbox_rtree import get_or_build_rtrees, nodes_in_bbox, node_mask_from_ids, filter_edges_by_node_masks

# default_response_class matches routes/inference.py. Without it FastAPI falls back to
# starlette's JSONResponse, whose render path is jsonable_encoder() -> json.dumps():
# the encoder walks the payload and builds a complete second copy of it, which on the
# ocean m2g response is 2,011,528 fresh dicts. Measured on that response, the two paths
# are 11.2 s (8.9 s encoder + 2.3 s dumps) against 0.25 s for orjson -- 45x, for a
# payload that is plain ints, floats and strings and needs none of the coercion.
#
# Routes declaring response_model= are unaffected: FastAPI still validates through
# jsonable_encoder regardless of the response class. Those return a handful of rows.
router = APIRouter(prefix="/graphs", tags=["Graphs"], default_response_class=ORJSONResponse)


_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
# _GRAPHS_BASE       = Path(__file__).parents[3] / "data" / "graphs"

# @router.get("/folders")
# def get_folders():
#     """Lists the available graph folder names (subdirs of the graphs base directory)."""
#     return list_graph_folders()




@router.get("", response_model=list[GraphMeta])
def list_graphs(db: Session = Depends(get_db)):
    """Lists all available graphs."""
    graphs = GraphCRUD.list_all(db)
    return graphs

@router.get("/{graph_id}", response_model=GraphMeta)
def get_graph(graph_id: str, db: Session = Depends(get_db)):
    """Get a specific graph by ID."""
    graph = GraphCRUD.get(db, graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph

SINGLE_SCALE_UPLOAD_FILES = {
    "grid_lat_lon.pt",
    "mesh_lat_lon.pt",
    "g2m_edge_index.pt",
    "g2m_features.pt",
    "m2m_edge_index.pt",
    "m2m_features.pt",
    "m2g_edge_index.pt",
    "m2g_features.pt",
}

HIERARCHICAL_UPLOAD_FILES = {
    "grid_lat_lon.pt",
    "mesh_lat_lon.pt",
    "g2m_edge_index.pt",
    "m2m_edge_index.pt",
    "m2g_edge_index.pt",
    "mesh_up_edge_index.pt",
    "mesh_down_edge_index.pt",
}

# Minimal neural-lam format (without features)
SINGLE_SCALE_MINIMAL_FILES = {
    "grid_lat_lon.pt",
    "mesh_lat_lon.pt",
    "g2m_edge_index.pt",
    "m2m_edge_index.pt",
    "m2g_edge_index.pt",
}


def _matches_upload_schema(saved_files: set[str], required_files: set[str]) -> bool:
    return required_files.issubset(saved_files)


def _is_anemoi_upload(saved_files: set[str]) -> bool:
    """Check if the upload is a single Anemoi graph.pt file."""
    return len(saved_files) == 1 and list(saved_files)[0] == "graph.pt"


def _detect_anemoi_hierarchy(graph_path: Path) -> bool:
    """
    Detect if an Anemoi graph is hierarchical (has multiple hidden layers).
    Returns True if hierarchical (multiple hidden layers), False if single-scale.
    """
    try:
        import torch
        graph = torch.load(graph_path, weights_only=False)
        
        # Check if it's a HeteroGraph with multiple hidden layers
        if hasattr(graph, 'node_types'):
            hidden_layers = [nt for nt in graph.node_types if nt.startswith('hidden_l')]
            return len(hidden_layers) > 1
        
        return False
    except Exception:
        # If we can't determine, assume single-scale (safe default)
        return False


def _parse_lcc(lcc: str | None) -> dict | None:
    """
    Parse the optional 'lat_0,lon_0[,lat_1,lat_2]' upload field into projection kwargs.

    Two values means the tangent case, i.e. a single standard parallel at lat_0.
    """
    if not lcc:
        return None
    try:
        parts = [float(v) for v in lcc.split(",")]
    except ValueError:
        raise ValueError(f"Could not parse lcc='{lcc}'. Expected comma-separated numbers.")
    if len(parts) == 2:
        parts += [parts[0], parts[0]]
    if len(parts) != 4:
        raise ValueError("lcc takes 2 or 4 comma-separated numbers: lat_0,lon_0[,lat_1,lat_2]")
    return dict(zip(("lat_0", "lon_0", "lat_1", "lat_2"), parts))


def _is_neural_lam_upload(saved_files: set[str]) -> bool:
    """Check if the upload is a neural-lam format (multiple files)."""
    # Check various neural-lam schemas
    return (
        _matches_upload_schema(saved_files, SINGLE_SCALE_UPLOAD_FILES)
        or _matches_upload_schema(saved_files, HIERARCHICAL_UPLOAD_FILES)
        or _matches_upload_schema(saved_files, SINGLE_SCALE_MINIMAL_FILES)
    )


@router.post("", response_model=GraphMeta)
async def create_graph(
    name: str = Form(...),
    description: str = Form(None),
    lcc: str = Form(None),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Creates a new graph by uploading .pt files.

    Supports three formats:
    1. Anemoi format: Single graph.pt file (HeteroGraph)
    2. Neural-lam format: Multiple .pt files (grid_lat_lon.pt, mesh_lat_lon.pt, etc.)
    3. Ocean format (Njord / baltic-sea-forecasting): edge indices + feature tables, with
       no position files. Converted by services.bsf_converter.

    Anemoi and ocean graphs are automatically converted to neural-lam format.

    Non-.pt files are ignored rather than rejected: ocean graph folders ship ~16 diagnostic
    .png plots alongside the tensors, and requiring users to weed them out by hand is a
    pointless step.

    Expected request:
    - name: human-readable name
    - description: optional description
    - lcc: optional 'lat_0,lon_0[,lat_1,lat_2]' Lambert Conformal Conic parameters, required
      only for limited-area ocean graphs whose coordinates are in projected metres
    - files: list of files to upload (non-.pt entries are skipped)
    """

    # Validate that we have files
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a unique hexID for graph directory
    random_hex = FileHandler.generate_random_hex()
    graph_dir = FileHandler.create_graph_directory(random_hex)
    
    try:
        saved_files = []
        skipped_files = []

        # Process each uploaded file
        for upload_file in files:
            # Skip anything that isn't a tensor (e.g. the .png plots shipped with ocean
            # graph folders, or .DS_Store). Rejecting the whole upload over them would make
            # these graphs unusable without manual pre-filtering.
            if not FileHandler.validate_file(upload_file.filename):
                skipped_files.append(upload_file.filename)
                continue

            # Save file to graph directory
            file_path = graph_dir / upload_file.filename
            await FileHandler.save_uploaded_file(upload_file, file_path)
            saved_files.append(upload_file.filename)

        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No .pt files in the upload"
                    + (f" ({len(skipped_files)} non-.pt files skipped)" if skipped_files else "")
                ),
            )

        saved_file_set = set(saved_files)
        
        # Detect format. Order matters: an ocean graph is identified by what it is MISSING
        # (the position files), so it must be checked before the neural-lam test, which
        # would simply reject it.
        is_anemoi = _is_anemoi_upload(saved_file_set)
        is_bsf = is_bsf_format(saved_file_set)
        is_neural_lam = _is_neural_lam_upload(saved_file_set)

        if is_bsf:
            try:
                projection = _parse_lcc(lcc)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            try:
                # In-place: the upload directory already holds the edge indices, and the
                # converter reads everything it needs before writing.
                convert_bsf_to_neural_lam(graph_dir, graph_dir, projection=projection)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to convert ocean graph: {e}"
                )
        elif is_anemoi:
            # Convert Anemoi to neural-lam format
            anemoi_file = graph_dir / "graph.pt"
            try:
                # Detect if the Anemoi graph is hierarchical (multiple hidden layers)
                is_hierarchical = _detect_anemoi_hierarchy(anemoi_file)
                convert_anemoi_to_neural_lam(anemoi_file, graph_dir, is_hierarchical=is_hierarchical)
                # Delete the original Anemoi file after conversion
                anemoi_file.unlink()
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to convert Anemoi graph: {str(e)}"
                )
        elif is_neural_lam:
            # Neural-lam format is already correct, proceed as-is
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported graph format. Upload either:\n"
                    "1. An Anemoi graph (single graph.pt file, with single or multiple hidden layers), or\n"
                    "2. A neural-lam graph (multiple .pt files: grid_lat_lon.pt, mesh_lat_lon.pt, "
                    "g2m_edge_index.pt, m2m_edge_index.pt, m2g_edge_index.pt, "
                    "mesh_up_edge_index.pt, mesh_down_edge_index.pt, etc.)"
                )
            )
        
        # Create database entry
        folder_path = FileHandler.get_graph_folder_path(random_hex)
        graph_create = GraphCreate(
            name=name,
            description=description,
            folder_path=folder_path
        )
        
        db_graph = GraphCRUD.create(db, graph_create)
        
        return db_graph
    
    except HTTPException:
        # Clean up directory if something went wrong
        FileHandler.delete_graph_directory(random_hex)
        raise
    except Exception as e:
        # Clean up directory on unexpected error
        FileHandler.delete_graph_directory(random_hex)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@router.put("/{graph_id}", response_model=GraphMeta)
def update_graph(graph_id: str, graph_update: GraphUpdate, db: Session = Depends(get_db)):
    """Updates graph metadata (name, description)."""
    graph = GraphCRUD.update(db, graph_id, graph_update)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph

@router.delete("/{graph_id}")
def delete_graph(graph_id: str, db: Session = Depends(get_db)):
    """Deletes a graph and its associated files."""
    # Delete from database
    success = GraphCRUD.delete(db, graph_id)
    if not success:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    # Delete files from filesystem
    FileHandler.delete_graph_directory(graph_id)
    
    return {"message": "Graph deleted successfully"}

# Retrieve the available graphs
# @router.get("")
# def get_graph(folder: str | None = None):
#     """
#     Returns the entire graph:
#       - grid_nodes : input data nodes on the grid
#       - mesh_nodes : latent graph nodes (where GNN operates)
#       - g2m_edges  : encoder edges (grid → mesh)
#       - m2m_edges  : GNN processing edges (mesh → mesh)
#       - m2g_edges  : decoder edges (mesh → grid)

#     Query parameter:
#       folder — graph folder name (e.g. "toy") or absolute path; omit for default.
#     """
#     try:
#         raw = load_graph(folder)
#         return {
#             "meta":       raw["meta"],
#             "grid_nodes": nodes_to_list(raw["grid_lat_lon"], "grid"),
#             "mesh_nodes": nodes_to_list(raw["mesh_lat_lon"], "mesh"),
#             "g2m_edges":  edges_to_list(raw["g2m_index"], raw["g2m_feats"]),
#             "m2m_edges":  edges_to_list(raw["m2m_index"], raw["m2m_feats"]),
#             "m2g_edges":  edges_to_list(raw["m2g_index"], raw["m2g_feats"]),
#         }
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=503, detail=str(e))

# Retrieve the available graphs
@router.get("")
def list_graphs():
    """
    Lists available graphs.
    For now, we only support one graph loaded from the default folder, so this will return a single entry if that graph exists.
    In the future, this could be extended to support multiple graphs with metadata.
    """
    with open(_DATA_DIR / "graphs" / "graphs.json") as f:
        return json.load(f)


@router.get("/{graph_id}/meta")
def get_graph_meta(graph_id: str, db: Session = Depends(get_db)):
    """Returns graph metadata including subgraph definitions (fast, no large payload)."""
    folder = GraphCRUD.get_path_by_id(db, graph_id)
    try:
        raw = load_graph(folder)
        _strip = {"edge_index", "src_lat_lon", "tgt_lat_lon"}
        subgraphs = [
            {
                "key": key,
                "n_edges": int(entry["edge_index"].shape[1]) if "edge_index" in entry else 0,
                **{k: v for k, v in entry.items() if k not in _strip},
            }
            for key, entry in raw["registry"].items()
        ]
        return {**raw["meta"], "subgraphs": subgraphs}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/{graph_id}/nodes/{node_type}")
def get_graph_nodes(
    graph_id: str,
    node_type: str,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    prev_min_lat: float | None = None,
    prev_max_lat: float | None = None,
    prev_min_lon: float | None = None,
    prev_max_lon: float | None = None,
    db: Session = Depends(get_db)
):
    """
    Returns nodes of a given type. Accepts any node type declared in the graph's
    metadata (e.g. "grid", "mesh" for single-scale; "grid", "mesh_0" … "mesh_3"
    for hierarchical). Pass min_lat/max_lat/min_lon/max_lon to restrict to the
    current map viewport. Pass prev_min_lat/... for delta mode (frontend merges).
    """
    folder = GraphCRUD.get_path_by_id(db, graph_id)
    try:
        raw = load_graph(folder)
        if node_type not in raw["meta"]["node_types"]:
            raise HTTPException(
                status_code=404,
                detail=f"Node type '{node_type}' not found. Available: {raw['meta']['node_types']}"
            )
        if node_type == "grid":
            lat_lon = raw["grid_lat_lon"]
        elif raw["meta"]["is_hierarchical"]:
            level = int(node_type.split("_")[1])
            lat_lon = raw["mesh_lat_lon"][level]
        else:
            lat_lon = raw["mesh_lat_lon"]
        # spatial bbox filtering — only apply if all 4 new bounds are provided
        if all(p is not None for p in [min_lat, max_lat, min_lon, max_lon]):
            has_prev = all(p is not None for p in [prev_min_lat, prev_max_lat, prev_min_lon, prev_max_lon])
            mask = bbox_node_mask(
                lat_lon, min_lat, max_lat, min_lon, max_lon,
                prev_min_lat if has_prev else None,
                prev_max_lat if has_prev else None,
                prev_min_lon if has_prev else None,
                prev_max_lon if has_prev else None,
            )
            ids = mask.nonzero(as_tuple=True)[0]
            return nodes_to_list(lat_lon[ids], node_type, ids=ids)
        return nodes_to_list(lat_lon, node_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/{graph_id}/edges/{subgraph_key}")
def get_edges(
    graph_id: str,
    subgraph_key: str,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    prev_min_lat: float | None = None,
    prev_max_lat: float | None = None,
    prev_min_lon: float | None = None,
    prev_max_lon: float | None = None,
    db: Session = Depends(get_db)
):
    """
    Returns edges for any subgraph key from the graph's registry
    (e.g. "g2m", "m2m", "m2m_0", "mesh_up_1", "m2g").
    Pass min_lat/max_lat/min_lon/max_lon to restrict to the current map viewport.
    Pass prev_min_lat/... as well to get only edges that just entered the viewport
    (delta mode — at least one endpoint is new; frontend merges rather than replaces).
    Edges are included if at least one endpoint falls inside the bbox (OR semantics).
    """
    folder = GraphCRUD.get_path_by_id(db, graph_id)
    try:
        raw = load_graph(folder)
        if subgraph_key not in raw["registry"]:
            raise HTTPException(
                status_code=404,
                detail=f"Subgraph key '{subgraph_key}' not found. Available: {sorted(raw['registry'].keys())}"
            )
        entry = raw["registry"][subgraph_key]
        ei          = entry["edge_index"]
        src_lat_lon = entry["src_lat_lon"]
        tgt_lat_lon = entry["tgt_lat_lon"]

        if all(p is not None for p in [min_lat, max_lat, min_lon, max_lon]):
            has_prev = all(p is not None for p in [prev_min_lat, prev_max_lat, prev_min_lon, prev_max_lon])
            src_mask_new = bbox_node_mask(src_lat_lon, min_lat, max_lat, min_lon, max_lon)
            tgt_mask_new = bbox_node_mask(tgt_lat_lon, min_lat, max_lat, min_lon, max_lon)
            src_mask_prev = bbox_node_mask(src_lat_lon, prev_min_lat, prev_max_lat, prev_min_lon, prev_max_lon) if has_prev else None
            tgt_mask_prev = bbox_node_mask(tgt_lat_lon, prev_min_lat, prev_max_lat, prev_min_lon, prev_max_lon) if has_prev else None
            ei, _ = filter_edges_by_node_masks(ei, None, src_mask_new, tgt_mask_new, src_mask_prev, tgt_mask_prev)

        return edges_to_list(ei, None, src_lat_lon, tgt_lat_lon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))



@router.get("/{graph_id}/graph_metrics/{subgraph_key}")
def get_graph_metrics_route(graph_id: str, subgraph_key: str, db: Session = Depends(get_db)):
    """
    Returns the full-graph degree and edge-length arrays for the given subgraph, as a binary
    container. Used by the frontend to build stable histograms that don't change on zoom.

    Binary rather than JSON because these are the largest fixed-size payloads the app fetches
    and they are refetched on every subgraph switch. On the ocean m2g subgraph the JSON form
    is 2,011,528 float lengths plus 675,219 target degrees -- 37.9 MB raw, 14.2 MB gzipped,
    ~1.6 s of `.tolist()` and orjson. The same content as typed arrays is 1.83 MB gzipped and
    serializes with `.numpy().tobytes()`. See LARGE_GRAPH_PERFORMANCE.md.

    Wire format -- little-endian throughout, which is every platform this runs on:

        offset 0        uint32   header_len
        offset 4        bytes    header_len bytes of UTF-8 JSON (schema below), space-padded
                                 so that 4 + header_len is a multiple of 4
        offset 4+H      bytes    the arrays, in header order, each padded up to a 4-byte
                                 offset so the client can view them in place

    The JSON header carries the scalars and describes each array:

        {
          "arrays": {
            "src_degrees": {"offset": 0,   "count": N_src, "dtype": "uint16"|"uint32"},
            "tgt_degrees": {"offset": ..., "count": N_tgt, "dtype": "uint16"|"uint32"},
            "lengths":     {"offset": ..., "count": E,     "dtype": "float32",
                            "sorted": true}
          },
          "degree_lo": float,      -- src mean - 2 sigma (suggested default slider lo)
          "degree_hi": float,      -- src mean + 2 sigma (suggested default slider hi)
          "tgt_degree_lo": float,  -- tgt mean - 2 sigma
          "tgt_degree_hi": float,
          "length_lo": float,
          "length_hi": float,
          "src_degree_min"/"_max", "tgt_degree_min"/"_max",
          "length_min"/"_max": float
        }

    Array offsets are relative to the start of the binary section (4 + header_len). Both that
    base and every array offset are multiples of 4, because the client builds uint16/uint32/
    float32 views straight over the response buffer and a TypedArray constructor throws on a
    byte offset that is not a multiple of its element size. Never emit an unpadded layout.

    `lengths` is sorted ascending. Every client-side use of it is a threshold query -- bin
    counts, "how many below the slider", discrete-level detection -- so sorted order turns
    each from an O(E) scan into a binary search. The min/max scalars are included so the
    client never has to scan for them either.

    Degree arrays stay per-node (not histogrammed server-side) because the map colours each
    node by its own degree; the dtype narrows to uint16 unless some node's degree exceeds
    65535, which no shipped graph comes close to (the widest is 4,694).
    """
    folder = GraphCRUD.get_path_by_id(db, graph_id)
    try:
        raw = load_graph(folder)
        if subgraph_key not in raw["registry"]:
            raise HTTPException(
                status_code=404,
                detail=f"Subgraph key '{subgraph_key}' not found. Available: {sorted(raw['registry'].keys())}"
            )
        m = get_graph_metrics(subgraph_key, folder)
        return _metrics_binary_response(m)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))


def _degree_bytes(deg: torch.Tensor) -> tuple[bytes, str]:
    """
    Packs a degree tensor to the narrowest unsigned type that holds it.

    uint16 covers every graph shipped or plausibly uploaded (max observed degree is 4,694 on
    ocean g2m), and halves the payload against uint32. The fallback is not dead code: a
    pathological hub node in an uploaded graph would silently wrap without it.
    """
    peak = int(deg.max()) if deg.numel() else 0
    if peak > 65535:
        return deg.to(torch.int64).numpy().astype("<u4").tobytes(), "uint32"
    return deg.to(torch.int64).numpy().astype("<u2").tobytes(), "uint16"


# The client builds typed-array views directly over the response buffer, and a TypedArray
# constructor rejects a byte offset that is not a multiple of its element size. 4 satisfies
# every dtype in the container (uint16 needs 2, uint32 and float32 need 4), so both the start
# of the binary section and each array offset within it are padded up to a multiple of 4.
#
# Without this the layout was whatever the header length and the node counts happened to
# produce: 12 of the 24 subgraphs across the shipped graphs landed unaligned, and every one of
# them threw a RangeError in the browser instead of drawing a distribution.
_ARRAY_ALIGN = 4


def _padding_to_align(offset: int, align: int = _ARRAY_ALIGN) -> int:
    """Bytes needed after `offset` to reach the next `align` boundary."""
    return -offset % align


def _metrics_binary_response(m: dict) -> Response:
    """Assembles the header + arrays container documented on get_graph_metrics_route."""
    src_deg, tgt_deg = m["src_deg"], m["tgt_deg"]
    lengths_sorted = m["lengths_sorted"]

    src_bytes, src_dtype = _degree_bytes(src_deg)
    tgt_bytes, tgt_dtype = _degree_bytes(tgt_deg)
    len_bytes = lengths_sorted.numpy().astype("<f4").tobytes()

    def _minmax(t: torch.Tensor) -> tuple[float, float]:
        if t.numel() == 0:
            return 0.0, 0.0
        return float(t.min()), float(t.max())

    src_min, src_max = _minmax(src_deg)
    tgt_min, tgt_max = _minmax(tgt_deg)
    # Sorted, so the extremes are the endpoints -- no scan.
    len_min = float(lengths_sorted[0]) if lengths_sorted.numel() else 0.0
    len_max = float(lengths_sorted[-1]) if lengths_sorted.numel() else 0.0

    # Lay the arrays out back to back, inserting padding so each one starts on an aligned
    # offset. The recorded offsets are relative to the start of the binary section.
    chunks: list[bytes] = []
    specs: dict[str, dict] = {}
    offset = 0
    for name, blob, dtype, count in (
        ("src_degrees", src_bytes, src_dtype, int(src_deg.numel())),
        ("tgt_degrees", tgt_bytes, tgt_dtype, int(tgt_deg.numel())),
        ("lengths",     len_bytes, "float32", int(lengths_sorted.numel())),
    ):
        pad = _padding_to_align(offset)
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad
        specs[name] = {"offset": offset, "count": count, "dtype": dtype}
        chunks.append(blob)
        offset += len(blob)
    specs["lengths"]["sorted"] = True

    header = {
        "arrays":         specs,
        "degree_lo":      _two_std_bounds(src_deg)[0],
        "degree_hi":      _two_std_bounds(src_deg)[1],
        "tgt_degree_lo":  _two_std_bounds(tgt_deg)[0],
        "tgt_degree_hi":  _two_std_bounds(tgt_deg)[1],
        "length_lo":      _two_std_bounds(m["lengths"])[0],
        "length_hi":      _two_std_bounds(m["lengths"])[1],
        "src_degree_min": src_min, "src_degree_max": src_max,
        "tgt_degree_min": tgt_min, "tgt_degree_max": tgt_max,
        "length_min":     len_min, "length_max":     len_max,
    }

    # Padded with spaces rather than NULs: the padding is inside the byte range the client
    # hands to JSON.parse, and whitespace is legal there. header_len counts the padding, so
    # the binary section starts at an aligned 4 + header_len.
    header_bytes = orjson.dumps(header)
    header_bytes += b" " * _padding_to_align(4 + len(header_bytes))

    body = struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(chunks)
    return Response(content=body, media_type="application/octet-stream")
