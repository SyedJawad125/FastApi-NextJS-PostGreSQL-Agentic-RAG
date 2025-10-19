from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

from app.core.config import settings

# Initialize embedding model
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

# FAISS index
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
documents: List[str] = []

def add_document_to_index(text: str):
    embeddings = embedding_model.encode([text])
    index.add(np.array(embeddings, dtype=np.float32))
    documents.append(text)

def search_similar_documents(query: str, k: int = 2):
    if len(documents) == 0:
        return []
    q_emb = embedding_model.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), k=k)
    return [documents[i] for i in I[0]]
