from pydantic import BaseModel
from datetime import datetime


class PromptRequest(BaseModel):
    prompt: str


class PromptResponse(BaseModel):
    url: str


class ImageDTO(BaseModel):
    id: str
    prompt: str
    save_path: str
    s3_url: str
    created_at: datetime