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
        if file is None:
            raise HTTPException(status_code=400, detail="File is required.")

        file_bytes = await file.read()
        filename = file.filename

        wrapper_path = generate_wrapper(
            task=task_type,
            model=model or "",
            custom_prompt=custom_prompt or "",
            filename=filename,
            file_bytes=file_bytes,
            token="your_access_token_here",
            api_url="http://localhost:8005/api/analyze/"
        )

        return {
            "message": f"Wrapper generated at {wrapper_path}",
            "task_type": task_type,
            "used_model": model,
            "response": "Sample response",
            "saved_files": {
                "request": {"txt": filename},
                "response": {"json": "output.json"}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
