from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class GraphMeta(BaseModel):
    graph_id: str
    name: str
    description: Optional[str] = None
    folder_path: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class GraphCreate(BaseModel):
    graph_id: str
    name: str
    description: Optional[str] = None
    folder_path: str

class GraphUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    folder_path: Optional[str] = None

class GraphUpload(BaseModel):
    """Schema for uploading a new graph."""
    graph_id: str
    name: str
    description: Optional[str] = None