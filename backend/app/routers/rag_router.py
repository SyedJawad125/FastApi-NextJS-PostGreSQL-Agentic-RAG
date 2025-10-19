from fastapi import APIRouter, UploadFile
from app.utils.pdf_reader import extract_text_from_pdf
from app.services.vectorstore import add_document_to_index, search_similar_documents
from app.services.groq_service import generate_answer_with_history
from app.services.memory_store import get_history, add_to_history
from app.models.schemas import AskRequest, AskResponse, UploadResponse
import uuid

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    """Upload and process a PDF file into the vector store."""
    text = extract_text_from_pdf(file.file)
    add_document_to_index(text)
    return {"message": "File successfully processed and added to knowledge base."}


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    """Ask a question with session-based memory and prompt template."""

    # Generate or reuse session_id
    session_id = payload.session_id or str(uuid.uuid4())

    # Retrieve chat history
    history = get_history(session_id)

    # Retrieve similar contexts
    contexts = search_similar_documents(payload.query, k=2)

    # Generate answer using LLM with prompt + memory
    answer = generate_answer_with_history(payload.query, contexts, history)

    # Save turn to memory
    add_to_history(session_id, payload.query, answer)

    # Return answer + session_id to keep conversation state
    return {"answer": answer, "session_id": session_id}
