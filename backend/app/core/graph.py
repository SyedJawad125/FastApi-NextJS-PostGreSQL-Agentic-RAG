from typing import Dict, List
from app.core.enums import GraphNodeType, GraphRelationType

class KnowledgeGraph:
    """Simple in-memory knowledge graph for Agentic RAG."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []

    def add_node(self, node_id: str, node_type: GraphNodeType, data: Dict = None):
        """Add a node to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type.value,
                "data": data or {}
            }

    def add_edge(self, source_id: str, target_id: str, relation: GraphRelationType):
        """Add a relation between nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append({
                "source": source_id,
                "target": target_id,
                "relation": relation.value
            })

    def get_node(self, node_id: str) -> Dict:
        """Retrieve a node by its ID."""
        return self.nodes.get(node_id)

    def get_relations(self, node_id: str) -> List[Dict]:
        """Retrieve all edges related to a node."""
        return [
            edge for edge in self.edges
            if edge["source"] == node_id or edge["target"] == node_id
        ]

# Example usage
if __name__ == "__main__":
    kg = KnowledgeGraph()

    # Add nodes
    kg.add_node("doc1", GraphNodeType.DOCUMENT, {"title": "Intro to RAG"})
    kg.add_node("chunk1", GraphNodeType.CHUNK, {"text": "RAG stands for Retrieval-Augmented Generation."})

    # Add relation
    kg.add_edge("doc1", "chunk1", GraphRelationType.CONTAINS)

    # Inspect
    print("Nodes:", kg.nodes)
    print("Edges:", kg.edges)
