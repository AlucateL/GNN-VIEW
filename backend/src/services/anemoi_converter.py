"""
Converter for Anemoi HeteroGraph format to neural-lam format.

Anemoi graphs are torch_geometric HeteroGraph objects with:

Single-scale format:
  - Node types: 'data' (grid points), 'hidden' (mesh/latent points)
  - Edge types: ('data', 'to', 'hidden'), ('hidden', 'to', 'data'), ('hidden', 'to', 'hidden')

Hierarchical format (multiple hidden layers):
  - Node types: 'data' (grid), 'hidden_l1', 'hidden_l2', ... 'hidden_lN' (mesh hierarchy levels)
  - Edge types: includes hierarchy edges like ('hidden_l1', 'to', 'hidden_l2')
  - Node attributes: 'pos' with latitude/longitude coordinates

Neural-lam format output:

Single-scale:
  - grid_lat_lon.pt: Grid node coordinates [N_grid, 2]
  - mesh_lat_lon.pt: [Mesh node coordinates] (single-element list) [[N_mesh, 2]]
  - g2m_edge_index.pt: Grid→Mesh edges [2, N_g2m]
  - m2m_edge_index.pt: [Mesh→Mesh edges] (single-element list)
  - m2g_edge_index.pt: Mesh→Grid edges [2, N_m2g]

Hierarchical:
  - grid_lat_lon.pt: Grid node coordinates [N_grid, 2]
  - mesh_lat_lon.pt: [coords_l1, coords_l2, ..., coords_lN] (list of tensors)
    - g2m_edge_index.pt: Grid→mesh layer adjacent to data [2, N_g2m]
  - m2m_edge_index.pt: [edges_l1, edges_l2, ..., edges_lN] (list of tensors)
    - m2g_edge_index.pt: mesh layer adjacent to data→Grid edges [2, N_m2g]
  - mesh_up_edge_index.pt: [edges_l0→l1, edges_l1→l2, ...] (hierarchy up)
  - mesh_down_edge_index.pt: [edges_l1→l0, edges_l2→l1, ...] (hierarchy down)

This converter extracts these components from an Anemoi HeteroGraph
and saves them in neural-lam format, auto-detecting single-scale vs hierarchical.
"""

from pathlib import Path
import math
import torch


def is_anemoi_format(file_path: Path) -> bool:
    """
    Check if a .pt file is in Anemoi HeteroGraph format.
    
    Returns True if the file contains a torch_geometric HeteroGraph
    with 'data' node type and either 'hidden' or 'hidden_l*' node types.
    """
    try:
        g = torch.load(file_path, weights_only=False)
        
        # Check if it's a HeteroGraph with expected node types
        if hasattr(g, 'node_types') and hasattr(g, 'edge_types'):
            node_types = set(g.node_types)
            # Anemoi format has 'data' and either 'hidden' (single-scale) or 'hidden_l*' (hierarchical)
            has_data = 'data' in node_types
            has_hidden = 'hidden' in node_types
            has_hidden_layers = any(nt.startswith('hidden_l') for nt in node_types)
            return has_data and (has_hidden or has_hidden_layers)
        return False
    except Exception:
        return False


def convert_anemoi_to_neural_lam(anemoi_graph_path: Path, output_dir: Path, is_hierarchical: bool = False) -> None:
    """
    Convert an Anemoi HeteroGraph .pt file to neural-lam format.
    
    Supports two formats:
    - Single-scale (is_hierarchical=False): outputs single mesh_lat_lon.pt
    - Hierarchical (is_hierarchical=True): outputs list of mesh tensors and hierarchy edges
    
    Args:
        anemoi_graph_path: Path to the Anemoi graph.pt file
        output_dir: Directory where neural-lam .pt files will be saved
        is_hierarchical: If True, treat graph as hierarchical with multiple hidden layers
        
    Raises:
        ValueError: If the graph doesn't have expected structure
        FileNotFoundError: If the input file doesn't exist
    """
    if not anemoi_graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {anemoi_graph_path}")
    
    print(f"Loading Anemoi graph from {anemoi_graph_path}...")
    print(f"  Mode: {'hierarchical' if is_hierarchical else 'single-scale'}")
    
    # Load Anemoi graph
    g = torch.load(anemoi_graph_path, weights_only=False)
    
    if not hasattr(g, 'node_types') or not hasattr(g, 'edge_types'):
        raise ValueError(
            "Input file is not a torch_geometric HeteroGraph. "
            "Expected Anemoi format with node_types and edge_types."
        )
    
    # Validate node types
    node_types = set(g.node_types)
    if 'data' not in node_types:
        raise ValueError(
            f"Unexpected node types: {g.node_types}. "
            "Expected 'data' (grid) and hidden layer nodes."
        )
    
    print(f"  Node types: {g.node_types}")
    print(f"  Edge types: {g.edge_types}")
    print(f"  Data nodes: {g['data'].num_nodes}")
    
    if is_hierarchical:
        _convert_anemoi_hierarchical(g, anemoi_graph_path, output_dir)
    else:
        _convert_anemoi_single_scale(g, anemoi_graph_path, output_dir)


def _convert_anemoi_single_scale(g, anemoi_graph_path: Path, output_dir: Path) -> None:
    """Convert single-scale Anemoi graph (single hidden layer) to neural-lam format."""
    # For single-scale, we expect a 'hidden' node type
    node_types = set(g.node_types)
    if 'hidden' not in node_types:
        raise ValueError(
            f"Single-scale mode requires 'hidden' node type, but found: {g.node_types}. "
            "Use is_hierarchical=True for multi-layer graphs."
        )
    
    print(f"  Hidden nodes: {g['hidden'].num_nodes}")
    
    # Extract positions from node attributes
    print("Extracting positions...")
    data_pos = _extract_positions(g['data'], node_type='data')
    hidden_pos = _extract_positions(g['hidden'], node_type='hidden')
    
    print(f"Before normalization:")
    print(f"  Data pos - lat: [{data_pos[:, 0].min():.2f}, {data_pos[:, 0].max():.2f}], lon: [{data_pos[:, 1].min():.2f}, {data_pos[:, 1].max():.2f}]")
    print(f"  Hidden pos - lat: [{hidden_pos[:, 0].min():.2f}, {hidden_pos[:, 0].max():.2f}], lon: [{hidden_pos[:, 1].min():.2f}, {hidden_pos[:, 1].max():.2f}]")
    
    # Normalize longitude from [0, 360] to [-180, 180] if needed
    data_pos = _normalize_longitude(data_pos)
    hidden_pos = _normalize_longitude(hidden_pos)
    
    print(f"After normalization:")
    print(f"  Data pos - lat: [{data_pos[:, 0].min():.2f}, {data_pos[:, 0].max():.2f}], lon: [{data_pos[:, 1].min():.2f}, {data_pos[:, 1].max():.2f}]")
    print(f"  Hidden pos - lat: [{hidden_pos[:, 0].min():.2f}, {hidden_pos[:, 0].max():.2f}], lon: [{hidden_pos[:, 1].min():.2f}, {hidden_pos[:, 1].max():.2f}]")
    
    # Extract edges
    print("Extracting edges...")
    # Anemoi edge types: ('data', 'to', 'hidden'), ('hidden', 'to', 'data'), ('hidden', 'to', 'hidden')
    g2m_edges = _extract_edge_index(g, ('data', 'to', 'hidden'))      # data → hidden
    m2g_edges = _extract_edge_index(g, ('hidden', 'to', 'data'))      # hidden → data
    m2m_edges = _extract_edge_index(g, ('hidden', 'to', 'hidden'))    # hidden → hidden
    
    print(f"  G2M edges: {g2m_edges.shape[1]}")
    print(f"  M2M edges: {m2m_edges.shape[1]}")
    print(f"  M2G edges: {m2g_edges.shape[1]}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in neural-lam format (single-scale)
    # For single-scale, wrap mesh tensors in single-element lists
    print(f"Saving single-scale neural-lam files to {output_dir}...")
    torch.save(data_pos, output_dir / "grid_lat_lon.pt")
    torch.save([hidden_pos], output_dir / "mesh_lat_lon.pt")
    torch.save(g2m_edges, output_dir / "g2m_edge_index.pt")
    torch.save([m2m_edges], output_dir / "m2m_edge_index.pt")
    torch.save(m2g_edges, output_dir / "m2g_edge_index.pt")
    
    # Create placeholder feature files (zeros) for compatibility with graph loading
    g2m_feats = torch.zeros((g2m_edges.shape[1], 0), dtype=torch.float32)
    m2m_feats = torch.zeros((m2m_edges.shape[1], 0), dtype=torch.float32)
    m2g_feats = torch.zeros((m2g_edges.shape[1], 0), dtype=torch.float32)
    
    torch.save(g2m_feats, output_dir / "g2m_features.pt")
    torch.save(m2m_feats, output_dir / "m2m_features.pt")
    torch.save(m2g_feats, output_dir / "m2g_features.pt")
    
    print("Single-scale conversion complete!")


def _convert_anemoi_hierarchical(g, anemoi_graph_path: Path, output_dir: Path) -> None:
    """Convert hierarchical Anemoi graph (multiple hidden layers) to neural-lam format."""
    # For hierarchical, collect all hidden_l* layers
    hidden_layers = sorted([nt for nt in g.node_types if nt.startswith('hidden_l')])
    
    if len(hidden_layers) < 2:
        raise ValueError(
            f"Hierarchical mode requires at least 2 hidden layers, but found: {hidden_layers}. "
            "Use is_hierarchical=False for single-layer graphs."
        )
    
    n_levels = len(hidden_layers)
    print(f"  Detected {n_levels} hidden layers: {hidden_layers}")
    
    # Extract grid positions
    print("Extracting positions...")
    data_pos = _extract_positions(g['data'], node_type='data')
    data_pos = _normalize_longitude(data_pos)
    print(f"  Grid pos - lat: [{data_pos[:, 0].min():.2f}, {data_pos[:, 0].max():.2f}], lon: [{data_pos[:, 1].min():.2f}, {data_pos[:, 1].max():.2f}]")
    
    # Extract all mesh level positions
    mesh_positions = []
    for layer_name in hidden_layers:
        pos = _extract_positions(g[layer_name], node_type=layer_name)
        pos = _normalize_longitude(pos)
        mesh_positions.append(pos)
        print(f"  {layer_name} pos - lat: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}], lon: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")
    
    # Extract edges
    print("Extracting edges...")
    
    # G2M/M2G source layers are not guaranteed to be the same on every Anemoi export.
    # Use the hidden layer that actually carries the data-adjacent edge type.
    g2m_edges = torch.zeros((2, 0), dtype=torch.long)
    g2m_source_layer = None
    for layer_name in hidden_layers:
        candidate = _extract_edge_index(g, ('data', 'to', layer_name))
        if candidate.shape[1] > 0:
            g2m_edges = candidate
            g2m_source_layer = layer_name
            break

    if g2m_source_layer is None:
        raise ValueError(
            "Could not find a data→hidden edge type in the hierarchical Anemoi graph. "
            f"Looked for ('data', 'to', {hidden_layers[0]!r}) through ('data', 'to', {hidden_layers[-1]!r})."
        )

    print(f"  G2M source layer: {g2m_source_layer}")
    print(f"  G2M edges: {g2m_edges.shape[1]}")
    
    m2g_edges = torch.zeros((2, 0), dtype=torch.long)
    m2g_source_layer = None
    for layer_name in hidden_layers:
        candidate = _extract_edge_index(g, (layer_name, 'to', 'data'))
        if candidate.shape[1] > 0:
            m2g_edges = candidate
            m2g_source_layer = layer_name
            break

    if m2g_source_layer is None:
        raise ValueError(
            "Could not find a hidden→data edge type in the hierarchical Anemoi graph. "
            f"Looked for ({hidden_layers[0]!r}, 'to', 'data') through ({hidden_layers[-1]!r}, 'to', 'data')."
        )

    print(f"  M2G source layer: {m2g_source_layer}")
    print(f"  M2G edges: {m2g_edges.shape[1]}")
    
    # M2M within each level
    m2m_indices = []
    for layer_name in hidden_layers:
        m2m_edges = _extract_edge_index(g, (layer_name, 'to', layer_name))
        m2m_indices.append(m2m_edges)
        print(f"  M2M ({layer_name}): {m2m_edges.shape[1]}")
    
    # Mesh hierarchy edges: up (lower → higher level) and down (higher → lower level)
    mesh_up_indices = []
    mesh_down_indices = []
    for i in range(len(hidden_layers) - 1):
        up_edges = _extract_edge_index(g, (hidden_layers[i], 'to', hidden_layers[i + 1]))
        down_edges = _extract_edge_index(g, (hidden_layers[i + 1], 'to', hidden_layers[i]))
        mesh_up_indices.append(up_edges)
        mesh_down_indices.append(down_edges)
        print(f"  Mesh up ({hidden_layers[i]} → {hidden_layers[i+1]}): {up_edges.shape[1]}")
        print(f"  Mesh down ({hidden_layers[i+1]} → {hidden_layers[i]}): {down_edges.shape[1]}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in neural-lam hierarchical format
    print(f"Saving hierarchical neural-lam files to {output_dir}...")
    torch.save(data_pos, output_dir / "grid_lat_lon.pt")
    torch.save(mesh_positions, output_dir / "mesh_lat_lon.pt")  # List of tensors
    torch.save(g2m_edges, output_dir / "g2m_edge_index.pt")
    torch.save(m2m_indices, output_dir / "m2m_edge_index.pt")    # List of tensors
    torch.save(m2g_edges, output_dir / "m2g_edge_index.pt")
    torch.save(mesh_up_indices, output_dir / "mesh_up_edge_index.pt")    # List of tensors
    torch.save(mesh_down_indices, output_dir / "mesh_down_edge_index.pt") # List of tensors
    
    print("Hierarchical conversion complete!")



def _extract_positions(node_data, node_type: str) -> torch.Tensor:
    """
    Extract latitude/longitude coordinates from a node's data.
    
    Tries multiple possible position sources:
    1. Direct 'pos' attribute (Anemoi standard)
    2. 'lat' and 'lon' attributes (alternative format)
    3. Fallback: zeros placeholder
    
    Returns [N, 2] tensor in [lat, lon] format.
    Anemoi typically uses [lon, lat] format, so we auto-detect and swap if needed.
    """
    pos = None
    
    # Try standard 'pos' attribute
    if hasattr(node_data, 'pos') and node_data.pos is not None:
        pos = node_data.pos
        if pos.shape[-1] >= 2:
            pos = pos[:, :2].clone()
    
    # Try separate lat/lon attributes
    elif (hasattr(node_data, 'lat') and node_data.lat is not None and
          hasattr(node_data, 'lon') and node_data.lon is not None):
        try:
            lat = node_data.lat.reshape(-1, 1)
            lon = node_data.lon.reshape(-1, 1)
            pos = torch.cat([lat, lon], dim=1)
        except Exception:
            pass
    
    # Try 'x' attribute which might be [N, 2] coordinates
    elif (hasattr(node_data, 'x') and node_data.x is not None and 
          node_data.x.shape[-1] >= 2):
        try:
            pos = node_data.x[:, :2].clone()
        except Exception:
            pass
    
    if pos is None:
        # Fallback: Create placeholder positions (zeros)
        num_nodes = node_data.num_nodes
        print(
            f"Warning: No position data found for {node_type} nodes. "
            f"Creating placeholder positions (zeros) for {num_nodes} nodes. "
            f"Expected 'pos', 'lat'/'lon', or 'x' attributes."
        )
        return torch.zeros(num_nodes, 2, dtype=torch.float32)

    pos = _maybe_convert_radians_to_degrees(pos, node_type)
    
    # Auto-detect format and swap if needed
    # Latitude must be in [-90, 90], longitude in [-360, 360]
    col0_min, col0_max = pos[:, 0].min().item(), pos[:, 0].max().item()
    col1_min, col1_max = pos[:, 1].min().item(), pos[:, 1].max().item()
    
    col0_abs_max = max(abs(col0_min), abs(col0_max))
    col1_abs_max = max(abs(col1_min), abs(col1_max))
    
    print(f"  {node_type} node positions - Col0: [{col0_min:.2f}, {col0_max:.2f}] (abs_max={col0_abs_max:.2f}), Col1: [{col1_min:.2f}, {col1_max:.2f}] (abs_max={col1_abs_max:.2f})")
    
    # Heuristic: latitude is always <= 90, longitude can be > 90
    col0_is_lat = col0_abs_max <= 90  # Latitude constraint
    col1_is_lat = col1_abs_max <= 90  # Latitude constraint
    
    # If column 1 looks like latitude (abs <= 90) and column 0 doesn't, swap
    if col1_is_lat and col1_abs_max <= 90 and col0_abs_max > 90:
        print(f"  → Detected [lon, lat] format, swapping to [lat, lon]")
        pos = pos[:, [1, 0]]  # Swap columns
    elif col0_is_lat or (col0_abs_max <= 90 and col1_abs_max > 90):
        print(f"  → Detected [lat, lon] format, using as-is")
    else:
        print(f"  → Ambiguous format, using as-is (assuming [lat, lon])")
    
    return pos


def _maybe_convert_radians_to_degrees(pos: torch.Tensor, node_type: str) -> torch.Tensor:
    """
    Convert radian coordinates to degrees when the tensor clearly looks like radians.

    Anemoi graph exports can store lat/lon in radians. The neural-lam loader
    expects degrees, so we only convert when the observed range fits a radian
    domain. This keeps already-degree inputs unchanged.
    """
    out = pos.clone().float()
    max_abs = max(out[:, 0].abs().max().item(), out[:, 1].abs().max().item())

    # Radian coordinates live roughly within [-pi/2, pi/2] for latitude and
    # [-2pi, 2pi] for longitude. Degree coordinates for global graphs are much larger.
    if max_abs <= (2 * math.pi + 0.25):
        print(f"  {node_type} positions look like radians; converting to degrees")
        out = torch.rad2deg(out)
    else:
        print(f"  {node_type} positions look like degrees; leaving as-is")

    return out


def _normalize_longitude(pos: torch.Tensor) -> torch.Tensor:
    """
    Normalize longitude from [0, 360] to [-180, 180] if needed.
    
    Args:
        pos: Tensor of shape [N, 2] with [lat, lon] (assumed after extraction/swapping)
        
    Returns:
        Normalized position tensor (longitude only, column 1)
    """
    out = pos.clone().float()
    
    # Column 1 should be longitude (after potential swap in _extract_positions)
    # Normalize any values > 180 by subtracting 360
    if out.shape[-1] >= 2:
        # Check if column 1 actually needs normalization (i.e., has values > 180)
        needs_norm = (out[:, 1] > 180).any()
        if needs_norm:
            print(f"  Normalizing longitude: col1 values > 180 detected")
            out[:, 1] = torch.where(
                out[:, 1] > 180,
                out[:, 1] - 360,
                out[:, 1]
            )
        else:
            print(f"  Longitude already in [-180, 180] range, no normalization needed")
    
    return out


def _extract_edge_index(g, edge_type: tuple) -> torch.Tensor:
    """
    Extract edge index tensor for a given edge type.
    
    Args:
        g: HeteroGraph object
        edge_type: Tuple of (src_type, relation, tgt_type)
        
    Returns:
        Edge index tensor of shape [2, num_edges]
    """
    if edge_type not in g.edge_types:
        # Edge type doesn't exist, return empty edge index
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = g[edge_type].edge_index
    
    # Ensure it's in the correct format [2, num_edges]
    if edge_index.shape[0] != 2:
        if edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        else:
            raise ValueError(
                f"Unexpected edge_index shape {edge_index.shape} for {edge_type}. "
                f"Expected [2, num_edges]."
            )
    
    return edge_index.clone()
