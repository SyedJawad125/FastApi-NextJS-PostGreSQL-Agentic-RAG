# """
# ===================================================================
# 4. app/services/rag_service.py - RAG Service Implementation
# ===================================================================
# """
# import os
# from typing import Dict, Any, List
# from groq import Groq
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# import pypdf
# import io


# class RAGService:
#     """Main RAG service for query processing"""
    
#     def __init__(self):
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#         self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
#         # Initialize ChromaDB
#         self.chroma_client = chromadb.PersistentClient(
#             path="./data/chromadb",
#             settings=Settings(anonymized_telemetry=False)
#         )
#         self.collection = self.chroma_client.get_or_create_collection(name="rag_documents")
    
#     async def simple_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
#         """Simple RAG: Retrieve + Generate"""
        
#         # Generate query embedding
#         query_embedding = self.embedding_model.encode(query).tolist()
        
#         # Search vector store
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         # Build context from retrieved documents
#         context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
        
#         # Generate answer using LLM
#         prompt = f"""Based on the following context, answer the question.

# Context:
# {context}

# Question: {query}

# Answer:"""
        
#         response = self.groq_client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=1000
#         )
        
#         answer = response.choices[0].message.content
        
#         return {
#             "answer": answer,
#             "chunks": [
#                 {"content": doc, "score": dist}
#                 for doc, dist in zip(results['documents'][0], results['distances'][0])
#             ] if results['documents'] else [],
#             "confidence": 0.85
#         }
    
#     async def agentic_query(self, query: str) -> Dict[str, Any]:
#         """Agentic RAG with reasoning"""
        
#         # Simple ReAct-style reasoning
#         reasoning_prompt = f"""Think step-by-step about how to answer this question: {query}

# Thought: Let me break this down...
# Action: Search for relevant information
# Observation: [Retrieved documents]
# Final Answer: [Synthesized answer]

# Provide a well-reasoned answer:"""
        
#         response = self.groq_client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": reasoning_prompt}],
#             temperature=0.7,
#             max_tokens=1500
#         )
        
#         answer = response.choices[0].message.content
        
#         return {
#             "answer": answer,
#             "chunks": [],
#             "confidence": 0.80
#         }
    
#     async def auto_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
#         """Auto-select best strategy"""
        
#         # Simple heuristic: use agentic for "why" and "how" questions
#         query_lower = query.lower()
        
#         if any(word in query_lower for word in ["why", "how", "explain", "analyze"]):
#             return await self.agentic_query(query)
#         else:
#             return await self.simple_query(query, top_k)
    
#     async def process_document(
#         self,
#         filename: str,
#         content: bytes,
#         content_type: str
#     ) -> Dict[str, Any]:
#         """Process and store document"""
        
#         # Extract text based on file type
#         if content_type == "application/pdf":
#             text = self._extract_pdf_text(content)
#         elif content_type == "text/plain":
#             text = content.decode('utf-8')
#         else:
#             text = content.decode('utf-8')
        
#         # Split into chunks
#         chunks = self._split_text(text, chunk_size=500)
        
#         # Generate embeddings
#         embeddings = self.embedding_model.encode(chunks).tolist()
        
#         # Store in vector database
#         ids = [f"{filename}_{i}" for i in range(len(chunks))]
#         metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
        
#         self.collection.add(
#             documents=chunks,
#             embeddings=embeddings,
#             ids=ids,
#             metadatas=metadatas
#         )
        
#         return {
#             "chunks_count": len(chunks),
#             "status": "success"
#         }
    
#     def _extract_pdf_text(self, content: bytes) -> str:
#         """Extract text from PDF"""
#         pdf_file = io.BytesIO(content)
#         reader = pypdf.PdfReader(pdf_file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
#         return text
    
#     def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
#         """Split text into chunks"""
#         words = text.split()
#         chunks = []
#         current_chunk = []
#         current_length = 0
        
#         for word in words:
#             current_chunk.append(word)
#             current_length += len(word) + 1
            
#             if current_length >= chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = []
#                 current_length = 0
        
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
        
#         return chunks





"""
===================================================================
app/services/rag_service.py - Working RAG Service Implementation
===================================================================
"""
import io
import logging
from typing import List, Dict, Any, Optional
import PyPDF2
from app.models.rag_model import DocumentChunk, GraphEntity, GraphRelationship, Session
import docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGService:
    """RAG Service with document processing and querying"""
    
    def __init__(self):
        """Initialize RAG service"""
        try:
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            data_dir = Path("data/vectors")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(data_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name="rag_documents")
                logger.info("✓ Using existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="rag_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("✓ Created new ChromaDB collection")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return content.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise Exception(f"Failed to extract text from TXT: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks if chunks else [text]
    
    async def process_document(
        self,
        filename: str,
        content: bytes,
        content_type: str
    ) -> Dict[str, Any]:
        """Process document and store in vector database"""
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text based on file type
            if content_type == "application/pdf":
                text = self.extract_text_from_pdf(content)
            elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.extract_text_from_docx(content)
            elif content_type == "text/plain":
                text = self.extract_text_from_txt(content)
            else:
                raise ValueError(f"Unsupported file type: {content_type}")
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty")
            
            logger.info(f"Extracted {len(text)} characters")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Generate embeddings and store in ChromaDB
            chunk_ids = [f"{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                ids=chunk_ids,
                metadatas=[{
                    "filename": filename,
                    "chunk_index": i,
                    "content_type": content_type
                } for i in range(len(chunks))]
            )
            
            logger.info(f"✓ Successfully stored {len(chunks)} chunks in vector database")
            
            return {
                "chunks_count": len(chunks),
                "method": "default",
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def simple_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Simple RAG query with retrieval and generation"""
        try:
            logger.info(f"Processing simple query: {query}")
            
            # Check if collection has any documents
            count = self.collection.count()
            if count == 0:
                return {
                    "answer": "No documents have been uploaded yet. Please upload documents first using the /upload endpoint.",
                    "chunks": [],
                    "confidence": 0.0
                }
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, count)
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    "answer": "No relevant information found in the uploaded documents.",
                    "chunks": [],
                    "confidence": 0.5
                }
            
            # Get retrieved chunks
            retrieved_docs = results['documents'][0]
            retrieved_distances = results['distances'][0] if results['distances'] else [0] * len(retrieved_docs)
            retrieved_metadata = results['metadatas'][0] if results['metadatas'] else [{}] * len(retrieved_docs)
            
            chunks = []
            for doc, dist, meta in zip(retrieved_docs, retrieved_distances, retrieved_metadata):
                chunks.append({
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "score": float(1 - dist),  # Convert distance to similarity score
                    "metadata": meta
                })
            
            # Generate answer from retrieved context
            context = "\n\n".join(retrieved_docs[:3])  # Use top 3 chunks
            
            # Simple answer generation (you can replace with LLM call)
            answer = self._generate_simple_answer(query, context)
            
            logger.info(f"✓ Query processed successfully")
            
            return {
                "answer": answer,
                "chunks": chunks,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Simple query failed: {e}")
            raise
    
    def _generate_simple_answer(self, query: str, context: str) -> str:
        """Generate a simple answer from context (placeholder for LLM)"""
        if not context or len(context.strip()) < 10:
            return "I couldn't find enough relevant information to answer your question."
        
        # For now, return a formatted response with context
        # In production, you'd call your LLM here
        return f"""Based on the available documents:

{context[:500]}...

Note: This is a direct excerpt from the documents. For more sophisticated answers, integrate with an LLM service."""
    
    async def agentic_query(self, query: str) -> Dict[str, Any]:
        """Agentic RAG query with reasoning"""
        # For now, fallback to simple query
        # You can implement ReAct pattern here later
        logger.info("Agentic query requested, using simple query for now")
        result = await self.simple_query(query, top_k=5)
        result["answer"] = f"[Agentic Mode]\n\n{result['answer']}"
        return result
    
    async def auto_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Auto-select best strategy"""
        # For now, use simple query
        # You can add strategy selection logic here
        logger.info("Auto strategy selected: simple")
        return await self.simple_query(query, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_chunks": 0,
                "collection_name": "unknown"
            }
    
    def reset(self):
        """Reset the collection (for testing)"""
        try:
            self.chroma_client.delete_collection(name="rag_documents")
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✓ Collection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

async def graph_rag_query(query_text: str, db: Session):
    """
    Use graph relationships to enhance context retrieval
    """
    # Step 1: Extract entities from query
    query_entities = await extract_entities_from_query(query_text)
    
    # Step 2: Find related entities from graph
    related_entities = []
    for entity_name in query_entities:
        # Find entity in graph
        entity = db.query(GraphEntity).filter(
            GraphEntity.name.ilike(f"%{entity_name}%")
        ).first()
        
        if entity:
            # Get connected entities via relationships
            relationships = db.query(GraphRelationship).filter(
                (GraphRelationship.source_id == entity.id) |
                (GraphRelationship.target_id == entity.id)
            ).all()
            
            for rel in relationships:
                related_entity = db.query(GraphEntity).filter(
                    GraphEntity.id.in_([rel.source_id, rel.target_id])
                ).all()
                related_entities.extend(related_entity)
    
    # Step 3: Get document chunks related to these entities
    relevant_chunks = []
    for entity in related_entities:
        doc_id = entity.properties.get("document_id")
        if doc_id:
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc_id
            ).all()
            relevant_chunks.extend(chunks)
    
    # Step 4: Combine with vector search results
    vector_chunks = await vector_search(query_text, db)
    
    # Step 5: Build enriched context
    context = build_graph_enriched_context(relevant_chunks, vector_chunks, related_entities)
    
    return context