from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..utils import wrapper_exists
from ..generator import generate_wrapper

router = APIRouter()

class WrapperRequest(BaseModel):
    task: str

@router.post("/generate-wrapper")
def generate_wrapper_endpoint(request: WrapperRequest):
    task = request.task

    if wrapper_exists(task):
        return {"message": f"Wrapper for '{task}' already exists."}

    try:
        path = generate_wrapper(task)
        return {"message": f"Wrapper generated at {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
