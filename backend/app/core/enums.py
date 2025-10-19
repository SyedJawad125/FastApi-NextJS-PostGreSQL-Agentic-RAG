from enum import Enum

class GraphNodeType(str, Enum):
    """Defines different types of nodes in the Agentic RAG graph."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"
    QUESTION = "question"
    ANSWER = "answer"
    TOOL = "tool"
    AGENT = "agent"

class GraphRelationType(str, Enum):
    """Defines different types of relationships between nodes in the graph."""
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    REFERENCES = "references"
    ANSWERS = "answers"
    USES = "uses"
    TRIGGERS = "triggers"
    SUPPORTS = "supports"
