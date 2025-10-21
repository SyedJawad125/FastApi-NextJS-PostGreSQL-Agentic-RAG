# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from typing import List

# from app.core.config import settings

# # Initialize embedding model
# embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

# # FAISS index
# dimension = embedding_model.get_sentence_embedding_dimension()
# index = faiss.IndexFlatL2(dimension)
# documents: List[str] = []

# def add_document_to_index(text: str):
#     embeddings = embedding_model.encode([text])
#     index.add(np.array(embeddings, dtype=np.float32))
#     documents.append(text)

# def search_similar_documents(query: str, k: int = 2):
#     if len(documents) == 0:
#         return []
#     q_emb = embedding_model.encode([query])
#     D, I = index.search(np.array(q_emb, dtype=np.float32), k=k)
#     return [documents[i] for i in I[0]]


# def get_count():
#     """Get total number of documents in vector store"""
#     return len(documents)

# def clear():
#     """Clear all documents and reset index"""
#     global index, documents
#     index.reset()
#     documents.clear()




# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from typing import List
# from app.core.config import settings

# class VectorStore:
#     def __init__(self):
#         self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
#         self.dimension = self.embedding_model.get_sentence_embedding_dimension()
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.documents: List[str] = []
#         self.metadata: List[dict] = []
    
#     def add_document(self, text: str, metadata: dict = None):
#         """Add a document to the index"""
#         embeddings = self.embedding_model.encode([text])
#         self.index.add(np.array(embeddings, dtype=np.float32))
#         self.documents.append(text)
#         self.metadata.append(metadata or {})
    
#     def search(self, query: str, k: int = 2):
#         """Search for similar documents"""
#         if len(self.documents) == 0:
#             return []
        
#         q_emb = self.embedding_model.encode([query])
#         D, I = self.index.search(np.array(q_emb, dtype=np.float32), k=k)
        
#         results = []
#         for idx in I[0]:
#             if idx < len(self.documents):
#                 results.append({
#                     "content": self.documents[idx],
#                     "metadata": self.metadata[idx],
#                     "score": float(D[0][len(results)])
#                 })
#         return results
    
#     def get_count(self):
#         """Get total number of documents"""
#         # Return the actual FAISS index count, not just the documents list
#         return self.index.ntotal
    
#     def clear(self):
#         """Clear all documents and reset index"""
#         # Reset FAISS index by removing all vectors
#         self.index.reset()
        
#         # Clear the documents and metadata lists
#         self.documents.clear()
#         self.metadata.clear()
        
#         count_after = self.get_count()
#         print(f"✅ Vector store cleared. Count: {count_after}")
        
#         # Verify it's actually cleared
#         if count_after != 0:
#             raise Exception(f"Failed to clear vector store: {count_after} vectors remain")
        
#         return True

# # Global singleton instance
# _vector_store_instance = None

# def get_vector_store():
#     """Get or create the singleton vector store instance"""
#     global _vector_store_instance
#     if _vector_store_instance is None:
#         _vector_store_instance = VectorStore()
#     return _vector_store_instance

# # Backward compatibility functions
# def add_document_to_index(text: str, metadata: dict = None):
#     return get_vector_store().add_document(text, metadata)

# def search_similar_documents(query: str, k: int = 2):
#     return get_vector_store().search(query, k)

# def get_count():
#     return get_vector_store().get_count()

# def clear():
#     return get_vector_store().clear()


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from app.core.config import settings

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[str] = []
        self.metadata: List[dict] = []
    
    def add_document(self, text: str, metadata: dict = None):
        """Add a document to the index"""
        embeddings = self.embedding_model.encode([text])
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.append(text)
        self.metadata.append(metadata or {})
    
    def search(self, query: str, k: int = 2):
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        q_emb = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), k=k)
        
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(D[0][len(results)])
                })
        return results
    
    def get_count(self):
        """Get total number of documents"""
        # Return the actual FAISS index count, not just the documents list
        return self.index.ntotal
    
    def clear(self):
        """Clear all documents and reset index"""
        # Reset FAISS index by removing all vectors
        self.index.reset()
        
        # Clear the documents and metadata lists
        self.documents.clear()
        self.metadata.clear()
        
        count_after = self.get_count()
        print(f"✅ Vector store cleared. Count: {count_after}")
        
        # Verify it's actually cleared
        if count_after != 0:
            raise Exception(f"Failed to clear vector store: {count_after} vectors remain")
        
        return True

# Global singleton instance
_vector_store_instance = None

def get_vector_store():
    """Get or create the singleton vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance

def reset_vector_store():
    """Reset the singleton instance (forces creation of new vector store)"""
    global _vector_store_instance
    _vector_store_instance = None
    print("🔄 Vector store singleton reset")

# Backward compatibility functions
def add_document_to_index(text: str, metadata: dict = None):
    return get_vector_store().add_document(text, metadata)

def search_similar_documents(query: str, k: int = 2):
    return get_vector_store().search(query, k)

def get_count():
    return get_vector_store().get_count()

def clear():
    return get_vector_store().clear()