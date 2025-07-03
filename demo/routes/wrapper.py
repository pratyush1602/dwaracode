from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from typing import Optional
from ..utils import wrapper_exists
from ..generator import generate_wrapper

router = APIRouter()

@router.post("/generate-wrapper")
async def generate_wrapper_endpoint(
    task_type: str = Form(...),
    model: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if wrapper_exists(task_type):
        return {"message": f"Wrapper for '{task_type}' already exists."}

    try:
        # You can access and process the file here as needed
        # contents = await file.read() if file else None
        path = generate_wrapper(task_type)
        return {
            "message": f"Wrapper generated at {path}",
            "task_type": task_type,
            "used_model": model,
            "response": "Sample response",  # Replace with real output
            "saved_files": {
                "request": {"txt": "path/to/input.txt"},
                "response": {"json": "path/to/output.json"}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
