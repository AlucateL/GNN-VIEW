from typing import Optional
from pydantic import BaseModel, field_validator
from datetime import datetime

from services.domains import DEFAULT_DOMAIN

class GraphMeta(BaseModel):
    graph_id: int
    name: str
    description: Optional[str] = None
    folder_path: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class GraphCreate(BaseModel):
    name: str
    description: Optional[str] = None
    folder_path: str

class GraphUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    folder_path: Optional[str] = None

class GraphUpload(BaseModel):
    """Schema for uploading a new graph."""
    name: str
    description: Optional[str] = None


class ModelMeta(BaseModel):
    model_id: int
    name: str
    description: Optional[str] = None
    model_path: str
    # Exposed so the frontend can tell a weather model from an ocean one -- e.g. to offer
    # only the graphs whose grid the model's output can be paired with.
    domain: str = DEFAULT_DOMAIN
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("domain", mode="before")
    @classmethod
    def _default_domain(cls, v):
        """
        Treat a missing domain as the default instead of failing validation.

        The column is NOT NULL with a server default, so this should not trigger -- but
        list_models validates every row into one response, so a single unmigrated row would
        turn the whole model list into a 500 and the picker would show nothing at all. A
        wrong-but-sane default on one row is a far better failure than an empty catalogue.
        """
        return DEFAULT_DOMAIN if v is None or v == "" else v

    class Config:
        from_attributes = True


class ModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    model_path: str
    domain: str = DEFAULT_DOMAIN


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    domain: Optional[str] = None