from pathlib import Path

from sqlalchemy.orm import Session
from models.models import Graph, Model
from models.schemas import GraphCreate, GraphUpdate, ModelCreate, ModelUpdate
from services.domains import normalize_domain
from datetime import datetime

class GraphCRUD:
    @staticmethod
    def create(db: Session, graph: GraphCreate) -> Graph:
        """Create a new graph."""
        db_graph = Graph(
            name=graph.name,
            description=graph.description,
            folder_path=graph.folder_path,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(db_graph)
        db.commit()
        db.refresh(db_graph)
        return db_graph

    @staticmethod
    def get(db: Session, graph_id: str) -> Graph:
        """Get a single graph by ID."""
        return db.query(Graph).filter(Graph.graph_id == int(graph_id)).first()

    @staticmethod
    def list_all(db: Session) -> list[Graph]:
        """List all graphs."""
        return db.query(Graph).all()

    @staticmethod
    def update(db: Session, graph_id: str, graph_update: GraphUpdate) -> Graph:
        """Update an existing graph."""
        db_graph = db.query(Graph).filter(Graph.graph_id == int(graph_id)).first()
        if not db_graph:
            return None
        
        update_data = graph_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_graph, key, value)
        
        db_graph.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_graph)
        return db_graph

    @staticmethod
    def delete(db: Session, graph_id: str) -> bool:
        """Delete a graph."""
        db_graph = db.query(Graph).filter(Graph.graph_id == int(graph_id)).first()
        if not db_graph:
            return False
        db.delete(db_graph)
        db.commit()
        return True
    
    @staticmethod
    def get_path_by_id(db: Session, graph_id: str) -> Path:
        """Helper to get the graph folder path by graph_id from the db."""
        db_graph = db.query(Graph).filter(Graph.graph_id == int(graph_id)).first()
        if not db_graph:
            return None
        return Path(db_graph.folder_path)


class ModelCRUD:
    @staticmethod
    def create(db: Session, model: ModelCreate) -> Model:
        """Create a new model."""
        db_model = Model(
            name=model.name,
            description=model.description,
            model_path=model.model_path,
            domain=normalize_domain(model.domain),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model

    @staticmethod
    def get(db: Session, model_id: str) -> Model:
        """Get a single model by ID."""
        return db.query(Model).filter(Model.model_id == int(model_id)).first()

    @staticmethod
    def list_all(db: Session) -> list[Model]:
        """List all models."""
        return db.query(Model).all()

    @staticmethod
    def update(db: Session, model_id: str, model_update: ModelUpdate) -> Model:
        """Update an existing model."""
        db_model = db.query(Model).filter(Model.model_id == int(model_id)).first()
        if not db_model:
            return None

        update_data = model_update.model_dump(exclude_unset=True)
        if "domain" in update_data:
            # Validate before assigning: an unrecognised domain would route the model to no
            # loader at all, and the failure would surface far from here.
            update_data["domain"] = normalize_domain(update_data["domain"])
        for key, value in update_data.items():
            setattr(db_model, key, value)

        db_model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_model)
        return db_model

    @staticmethod
    def delete(db: Session, model_id: str) -> bool:
        """Delete a model."""
        db_model = db.query(Model).filter(Model.model_id == int(model_id)).first()
        if not db_model:
            return False
        db.delete(db_model)
        db.commit()
        return True

    @staticmethod
    def get_by_filename(db: Session, filename: str) -> Model | None:
        """Find a model whose model_path ends with the given filename."""
        return db.query(Model).filter(Model.model_path.like(f"%{filename}")).first()