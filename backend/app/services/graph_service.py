# # In services/chunking.py or a new graph_service.py

# import json
# from django.conf import settings
# from app.models.schemas import GraphEntity, GraphRelationship
# from sqlalchemy.orm import Session
# import uuid

# async def extract_graph_from_chunk(chunk_content: str, document_id: str, db: Session):
#     """
#     Extract entities and relationships from a chunk using your Groq LLM
#     """
#     prompt = f"""
#     Extract entities and their relationships from the following text.
    
#     Text: {chunk_content}
    
#     Return a JSON with:
#     {{
#         "entities": [
#             {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT", "description": "brief description"}}
#         ],
#         "relationships": [
#             {{"source": "entity1", "target": "entity2", "relation": "relationship_type", "weight": 0.8}}
#         ]
#     }}
#     """
    
#     # Call your Groq LLM (similar to your existing query service)
#     response = await call_groq_llm(prompt)
#     graph_data = parse_json_response(response)
    
#     # Store entities
#     entity_map = {}
#     for entity_data in graph_data.get("entities", []):
#         entity = GraphEntity(
#             id=str(uuid.uuid4()),
#             name=entity_data["name"],
#             type=entity_data["type"],
#             description=entity_data.get("description", ""),
#             properties={"document_id": document_id}
#         )
#         db.add(entity)
#         entity_map[entity.name] = entity.id
    
#     # Store relationships
#     for rel_data in graph_data.get("relationships", []):
#         if rel_data["source"] in entity_map and rel_data["target"] in entity_map:
#             relationship = GraphRelationship(
#                 id=str(uuid.uuid4()),
#                 source_id=entity_map[rel_data["source"]],
#                 target_id=entity_map[rel_data["target"]],
#                 relation_type=rel_data["relation"],
#                 weight=rel_data.get("weight", 1.0),
#                 properties={"document_id": document_id}
#             )
#             db.add(relationship)
    
#     db.commit()



# from groq import Groq

# async def extract_entities_with_groq(text: str):
#     """
#     Use Groq to extract entities and relationships
#     """
#     client = Groq(api_key=settings.GROQ_API_KEY)
    
#     prompt = f"""
#     Extract all named entities and their relationships from this text.
#     Format as JSON with entities and relationships arrays.
    
#     Text: {text}
#     """
    
#     response = client.chat.completions.create(
#         model="mixtral-8x7b-32768",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1,
#         response_format={"type": "json_object"}
#     )
    
#     return json.loads(response.choices[0].message.content)





"""
Graph extraction and management service
Extracts entities and relationships from text using Groq LLM
"""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.rag_model import GraphEntity, GraphRelationship
from groq import Groq
import json
import uuid
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️  GROQ_API_KEY not set, graph extraction will fail")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


async def extract_entities_from_text(text: str) -> Dict[str, Any]:
    """
    Extract entities and relationships from text using Groq LLM
    
    Args:
        text: Text content to analyze
    
    Returns:
        Dictionary with entities and relationships
    """
    if not groq_client:
        logger.error("❌ Groq client not initialized")
        return {"entities": [], "relationships": []}
    
    try:
        prompt = f"""Extract all named entities and their relationships from the following text.
        
Return ONLY a valid JSON object with this exact structure:
{{
    "entities": [
        {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|TECHNOLOGY|EVENT", "description": "brief description"}}
    ],
    "relationships": [
        {{"source": "entity1_name", "target": "entity2_name", "relation": "relationship_type", "weight": 0.8}}
    ]
}}

Rules:
- Extract only meaningful, specific entities (proper nouns, technical terms, important concepts)
- Types: PERSON (people), ORGANIZATION (companies, institutions), LOCATION (places), CONCEPT (ideas, theories), TECHNOLOGY (tools, systems), EVENT (specific events)
- Relationships should be clear and meaningful (e.g., "WORKS_AT", "LOCATED_IN", "FOUNDED_BY", "USES", "PART_OF")
- Weight should be 0.5-1.0 based on relationship strength
- Keep descriptions concise (under 50 words)

Text to analyze:
{text[:2000]}  

Return only the JSON, no markdown formatting."""

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        result = json.loads(content)
        
        # Validate structure
        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
        
        logger.info(f"✅ Extracted {len(result['entities'])} entities, {len(result['relationships'])} relationships")
        
        return result
        
    except json.JSONDecodeError as je:
        logger.error(f"❌ Failed to parse JSON from Groq response: {str(je)}")
        logger.debug(f"Response content: {content}")
        return {"entities": [], "relationships": []}
    except Exception as e:
        logger.error(f"❌ Entity extraction failed: {str(e)}", exc_info=True)
        return {"entities": [], "relationships": []}


async def extract_graph_from_document(
    document_id: str,
    chunks: List[str],
    db: Session,
    max_chunks: int = 10  # Limit to avoid excessive API calls
) -> Dict[str, Any]:
    """
    Extract knowledge graph from document chunks
    
    Args:
        document_id: Document ID to associate entities with
        chunks: List of text chunks from the document
        db: Database session
        max_chunks: Maximum number of chunks to process
    
    Returns:
        Dictionary with extraction statistics
    """
    entity_map = {}  # name -> entity_id mapping
    all_entities = []
    all_relationships = []
    
    logger.info(f"🕸️  Starting graph extraction for document {document_id}")
    
    # Process chunks (limit to avoid excessive API calls)
    chunks_to_process = chunks[:max_chunks]
    logger.info(f"📊 Processing {len(chunks_to_process)} of {len(chunks)} chunks")
    
    for idx, chunk_text in enumerate(chunks_to_process):
        if len(chunk_text.strip()) < 50:  # Skip very short chunks
            continue
        
        logger.info(f"  Processing chunk {idx + 1}/{len(chunks_to_process)}...")
        
        # Extract entities and relationships from chunk
        graph_data = await extract_entities_from_text(chunk_text)
        
        # Process entities
        for entity_data in graph_data.get("entities", []):
            entity_name = entity_data.get("name", "").strip()
            
            if not entity_name or len(entity_name) < 2:
                continue
            
            # Check if entity already exists (case-insensitive)
            entity_name_lower = entity_name.lower()
            
            if entity_name_lower not in entity_map:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity = GraphEntity(
                    id=entity_id,
                    name=entity_name,
                    type=entity_data.get("type", "CONCEPT"),
                    description=entity_data.get("description", ""),
                    properties={
                        "document_id": document_id,
                        "chunk_index": idx,
                        "source": "groq_extraction"
                    }
                )
                db.add(entity)
                entity_map[entity_name_lower] = entity_id
                all_entities.append(entity)
            else:
                # Update existing entity description if better
                existing_id = entity_map[entity_name_lower]
                existing = db.query(GraphEntity).filter(
                    GraphEntity.id == existing_id
                ).first()
                
                new_desc = entity_data.get("description", "")
                if existing and len(new_desc) > len(existing.description or ""):
                    existing.description = new_desc
        
        # Process relationships
        for rel_data in graph_data.get("relationships", []):
            source_name = rel_data.get("source", "").strip().lower()
            target_name = rel_data.get("target", "").strip().lower()
            
            if source_name in entity_map and target_name in entity_map:
                relationship = GraphRelationship(
                    id=str(uuid.uuid4()),
                    source_id=entity_map[source_name],
                    target_id=entity_map[target_name],
                    relation_type=rel_data.get("relation", "RELATED_TO"),
                    weight=float(rel_data.get("weight", 1.0)),
                    properties={
                        "document_id": document_id,
                        "chunk_index": idx,
                        "source": "groq_extraction"
                    }
                )
                db.add(relationship)
                all_relationships.append(relationship)
    
    # Commit all changes
    try:
        db.commit()
        logger.info(f"✅ Graph extraction complete: {len(all_entities)} entities, {len(all_relationships)} relationships")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to save graph: {str(e)}")
        raise
    
    return {
        "entities_count": len(all_entities),
        "relationships_count": len(all_relationships),
        "chunks_processed": len(chunks_to_process)
    }


def get_document_graph(document_id: str, db: Session) -> Dict[str, Any]:
    """
    Retrieve knowledge graph for a specific document
    
    Args:
        document_id: Document ID
        db: Database session
    
    Returns:
        Graph structure with nodes and edges
    """
    # Get all entities for this document
    entities = db.query(GraphEntity).filter(
        GraphEntity.properties['document_id'].astext == document_id
    ).all()
    
    if not entities:
        logger.warning(f"⚠️  No entities found for document {document_id}")
        return {"nodes": [], "edges": []}
    
    entity_ids = [e.id for e in entities]
    
    # Get all relationships between these entities
    relationships = db.query(GraphRelationship).filter(
        GraphRelationship.source_id.in_(entity_ids),
        GraphRelationship.target_id.in_(entity_ids)
    ).all()
    
    # Format for frontend
    nodes = [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "description": e.description or ""
        }
        for e in entities
    ]
    
    edges = [
        {
            "source": r.source_id,
            "target": r.target_id,
            "relation": r.relation_type,
            "weight": r.weight
        }
        for r in relationships
    ]
    
    logger.info(f"📊 Retrieved graph: {len(nodes)} nodes, {len(edges)} edges")
    
    return {
        "nodes": nodes,
        "edges": edges
    }