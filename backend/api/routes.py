from fastapi import APIRouter, Depends
from schemas.job_post import JobPost, InterviewResponse
from services.interview_service import InterviewService

router = APIRouter()

@router.post("/generate_interview/{model_index}", response_model=InterviewResponse)
async def generate_interview(model_index: int, job_post: JobPost, service: InterviewService=Depends(InterviewService)):
    return service.generate_interview_qa(job_post, model_index)