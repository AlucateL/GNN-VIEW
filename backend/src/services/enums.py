from enum import Enum

class NodeType(str, Enum):
    GRID = "grid"
    MESH = "mesh"

class EdgeType(str, Enum):
    G2M = "g2m"
    M2M = "m2m"
    M2G = "m2g"

