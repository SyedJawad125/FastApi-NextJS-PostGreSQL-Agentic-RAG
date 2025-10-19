"""
app/schemas/graph_schemas.py - Graph-specific schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.core.enums import GraphNodeType, GraphRelationType


class EntityCreate(BaseModel):
    """Create entity request"""
    name: str
    type: GraphNodeType
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntityResponse(BaseModel):
    """Entity response"""
    id: str
    name: str
    type: GraphNodeType
    description: str
    properties: Dict[str, Any]
    created_at: Optional[str] = None


class RelationshipCreate(BaseModel):
    """Create relationship request"""
    source_id: str
    target_id: str
    relation_type: GraphRelationType
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationshipResponse(BaseModel):
    """Relationship response"""
    id: str
    source_id: str
    target_id: str
    relation_type: GraphRelationType
    properties: Dict[str, Any]
    created_at: Optional[str] = None


class GraphStats(BaseModel):
    """Knowledge graph statistics"""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    connected_components: int
    average_degree: float
    density: float


class SubgraphRequest(BaseModel):
    """Request for subgraph extraction"""
    entity_ids: List[str]
    max_depth: int = Field(default=2, ge=1, le=5)
    include_properties: bool = True


class SubgraphResponse(BaseModel):
    """Subgraph response"""
    nodes: List[EntityResponse]
    edges: List[RelationshipResponse]
    metadata: Dict[str, Any]


class PathQuery(BaseModel):
    """Query for path between entities"""
    source_id: str
    target_id: str
    max_length: int = Field(default=5, ge=1, le=10)


class PathResponse(BaseModel):
    """Path between entities"""
    paths: List[List[str]]
    shortest_path_length: Optional[int] = None
    total_paths_found: int


class GraphVisualization(BaseModel):
    """Graph visualization data"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str = "force"
    metadata: Dict[str, Any] = Field(default_factory=dict)