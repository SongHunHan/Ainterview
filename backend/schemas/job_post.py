from pydantic import BaseModel

class JobPost(BaseModel):
    content: str

class InterviewResponse(BaseModel):
    question: str
    answer: str