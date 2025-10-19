from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
