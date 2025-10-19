"""
===================================================================
4. app/services/rag_service.py - RAG Service Implementation
===================================================================
"""
import os
from typing import Dict, Any, List
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf
import io


class RAGService:
    """Main RAG service for query processing"""
    
    def __init__(self):
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chromadb",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(name="rag_documents")
    
    async def simple_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Simple RAG: Retrieve + Generate"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Build context from retrieved documents
        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "chunks": [
                {"content": doc, "score": dist}
                for doc, dist in zip(results['documents'][0], results['distances'][0])
            ] if results['documents'] else [],
            "confidence": 0.85
        }
    
    async def agentic_query(self, query: str) -> Dict[str, Any]:
        """Agentic RAG with reasoning"""
        
        # Simple ReAct-style reasoning
        reasoning_prompt = f"""Think step-by-step about how to answer this question: {query}

Thought: Let me break this down...
Action: Search for relevant information
Observation: [Retrieved documents]
Final Answer: [Synthesized answer]

Provide a well-reasoned answer:"""
        
        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "chunks": [],
            "confidence": 0.80
        }
    
    async def auto_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Auto-select best strategy"""
        
        # Simple heuristic: use agentic for "why" and "how" questions
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["why", "how", "explain", "analyze"]):
            return await self.agentic_query(query)
        else:
            return await self.simple_query(query, top_k)
    
    async def process_document(
        self,
        filename: str,
        content: bytes,
        content_type: str
    ) -> Dict[str, Any]:
        """Process and store document"""
        
        # Extract text based on file type
        if content_type == "application/pdf":
            text = self._extract_pdf_text(content)
        elif content_type == "text/plain":
            text = content.decode('utf-8')
        else:
            text = content.decode('utf-8')
        
        # Split into chunks
        chunks = self._split_text(text, chunk_size=500)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Store in vector database
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        return {
            "chunks_count": len(chunks),
            "status": "success"
        }
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
