from pathlib import Path

from sqlalchemy import text

from .database import engine, SessionLocal
from models.models import Base, Graph, Model
from services.crud import GraphCRUD, ModelCRUD
from models.schemas import GraphCreate, ModelCreate
from services.domains import DEFAULT_DOMAIN, DOMAIN_ERA5, DOMAIN_OCEAN
from services.graph import _GRAPHS_BASE

def _migrate_model_domain() -> None:
    """
    Add models.domain to databases created before the column existed.

    create_all() only creates missing *tables*, never missing columns, so an existing
    deployment would keep a domain-less models table and every query naming the column would
    fail. Written as raw idempotent DDL because the project has no migration tool; the
    server_default backfills existing rows to 'era5', which is correct -- ERA5 was the only
    domain that worked before this column.
    """
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE models ADD COLUMN IF NOT EXISTS domain "
            f"VARCHAR NOT NULL DEFAULT '{DEFAULT_DOMAIN}'"
        ))


def init_db():
    """Create tables and seed initial data."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    _migrate_model_domain()

    db = SessionLocal()
    
    # Seed default graphs. Presets are ensured by folder_path rather than only on an empty
    # database, so adding one later reaches existing deployments too — otherwise a new preset
    # is invisible to anyone whose DB was already seeded.
    default_graphs = [
        GraphCreate(
            name="Graphcast-like GNN",
            folder_path="/usr/data/graphs/global_multiscale",
            description="Graphcast-like GNN"
        ),
        GraphCreate(
            name="Hierarchical GNN (4-levels)",
            folder_path="/usr/data/graphs/global_hierarchical",
            description="Hierarchical GNN (4-levels)"
        ),
        GraphCreate(
            name="Ocean Hierarchical GNN (3-levels)",
            folder_path="/usr/data/graphs/ocean_hierarchical",
            description="Global ocean analysis graph — grid nodes are sea points only"
        ),
        GraphCreate(
            name="Ocean Hierarchical GNN (3-levels, cluster-based)",
            folder_path="/usr/data/graphs/ocean_cluster",
            description=(
                "Global ocean analysis graph with a k-means clustered mesh — the same sea "
                "grid as the icosahedral preset above, meshed by clustering instead"
            )
        ),
    ]

    for graph_data in default_graphs:
        existing = db.query(Graph).filter(
            Graph.folder_path == graph_data.folder_path
        ).first()
        if existing:
            continue
        GraphCRUD.create(db, graph_data)
        print(f"Created graph: {graph_data.name}")

    # Drop rows whose folder no longer exists — e.g. an uploaded graph that has since
    # been renamed into a preset. Without this they linger in the picker and 404.
    # Preset paths are kept even when the PVC is empty so a first boot before the
    # manual data upload does not wipe the picker.
    _preset_graph_paths = {g.folder_path for g in default_graphs}
    for graph in db.query(Graph).all():
        if graph.folder_path in _preset_graph_paths:
            continue
        local = Path("/usr/data/graphs") / Path(graph.folder_path).name
        repo = _GRAPHS_BASE / Path(graph.folder_path).name
        if not local.exists() and not repo.exists():
            print(f"Removing graph with missing folder: {graph.name} ({graph.folder_path})")
            db.delete(graph)
    db.commit()

    # Rename legacy preset entries if they still have the old names.
    renames = {
        "/usr/data/graphs/global_multiscale": "Graphcast-like GNN",
        "/usr/data/graphs/global_hierarchical": "Hierarchical GNN (4-levels)",
        "/usr/data/graphs/ocean_hierarchical": "Ocean Hierarchical GNN (3-levels)",
    }
    for folder_path, new_name in renames.items():
        graph = db.query(Graph).filter(Graph.folder_path == folder_path).first()
        if graph and graph.name != new_name:
            graph.name = new_name
            graph.description = new_name
            db.commit()
            print(f"Renamed graph at {folder_path} → {new_name}")

    # Seed explain models. Ensured by model_path rather than only on an empty table, for the
    # same reason the graph presets above are: gating on `count() == 0` means any preset added
    # later is invisible to every database that was already seeded. That is exactly what
    # happened to SeaCast -- it joined this list well after the first deployments, so it never
    # appeared in the model picker on any existing install.
    default_models = [
        ModelCreate(
            name="GraphCast Scripted",
            model_path="/usr/data/paper_checkpoints/graphcast_scripted.pt",
            description="TorchScript export of GraphCast used in Explain mode",
            domain=DOMAIN_ERA5,
        ),
        ModelCreate(
            name="Graph FM Scripted (Hierarchical)",
            model_path="/usr/data/paper_checkpoints/graph_fm_scripted.pt",
            description="TorchScript export of hierarchical Graph FM used in Explain mode",
            domain=DOMAIN_ERA5,
        ),
        ModelCreate(
            name="SeaCast Scripted (Ocean)",
            model_path="/usr/data/paper_checkpoints/seacast_scripted.pt",
            description="TorchScript export of SeaCast; pairs with the ocean graph and the OceanBench dataset",
            domain=DOMAIN_OCEAN,
        ),
    ]

    for model_data in default_models:
        existing = db.query(Model).filter(
            Model.model_path == model_data.model_path
        ).first()
        if existing:
            continue
        ModelCRUD.create(db, model_data)
        print(f"Created model: {model_data.name}")

    # Correct the domain on preset rows that predate the column: the ALTER TABLE backfills
    # every existing row to 'era5', which is right for the two weather models and wrong for
    # SeaCast. Keyed on model_path -- it is what the loader actually uses, and unlike the
    # display name it is not something an operator renames.
    model_domains = {m.model_path: m.domain for m in default_models}
    for model_path, domain in model_domains.items():
        model = db.query(Model).filter(Model.model_path == model_path).first()
        if model and model.domain != domain:
            print(f"Corrected domain for {model.name}: {model.domain} → {domain}")
            model.domain = domain
    db.commit()

    db.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()