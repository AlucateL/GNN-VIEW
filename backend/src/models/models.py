from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from services.domains import DEFAULT_DOMAIN

Base = declarative_base()

class Graph(Base):
    __tablename__ = "graphs"

    graph_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    folder_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Model(Base):
    __tablename__ = "models"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    model_path = Column(String, nullable=False)
    # Which dataset + loader this model runs against. See services/domains.py -- an ocean
    # model loaded through the ERA5 loader returns a module that cannot be called.
    # server_default (not just default) so the ALTER TABLE in seed_db backfills existing rows.
    domain = Column(String, nullable=False, server_default=DEFAULT_DOMAIN, default=DEFAULT_DOMAIN)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)