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

class AgentType(str, Enum):
    """Defines different types of agents."""
    SYSTEM = "system"
    USER = "user"

from enum import Enum

# class RAGStrategy(str, Enum):
#     """Defines different RAG strategies."""
#     STANDARD = "standard"
#     HYBRID = "hybrid"
#     ADVANCED = "advanced"

#     SIMPLE = "simple"
#     AGENTIC = "agentic" 
#     GRAPH = "graph"
#     MULTI_AGENT = "multi_agent"
#     HYBRID = "hybrid"
#     AUTO = "auto"

class RAGStrategy(Enum):
    SIMPLE = "simple"
    AGENTIC = "agentic" 
    GRAPH = "graph"
    MULTI_AGENT = "multi_agent"
    HYBRID = "hybrid"
    AUTO = "auto"


class QueryType(str, Enum):
    """Defines different types of queries for Agentic RAG."""
    USER_QUERY = "user_query"
    SYSTEM_QUERY = "system_query"


class ToolType(str, Enum):
    """Defines different types of tools in the Agentic RAG system."""
    SEARCH = "search"
    RETRIEVAL = "retrieval"
    SUMMARIZER = "summarizer"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
