from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Request
from typing import Optional
from demo.utils import wrapper_exists, get_wrapper_path
from demo.generator import generate_wrapper
from pathlib import Path
import importlib.util
import os
import traceback

# print("hello there")
router = APIRouter()

@router.post("/generate-wrapper")
async def generate_wrapper_endpoint(
    request: Request,
    task_type: str = Form(...)
):
    # Retrieve session ID from cookies for consistency
    session_id = request.cookies.get("api_key_session")
    if not session_id:
        # Optionally raise an error if session ID is required
        # raise HTTPException(status_code=401, detail="No active session found")
        pass  # Currently, session ID is not strictly required for wrapper generation
    
    # print(session_id)

    if wrapper_exists(task_type):
        return {"message": f"Wrapper for '{task_type}' already exists."}

    try:
        path = generate_wrapper(task_type)
        return {
            "message": f"Wrapper generated at {path}",
            "task_type": task_type,
            "session_id": session_id  # Include session ID in response for debugging
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wrapper generation failed: {str(e)}")


@router.post("/wrappers/{task}_wrapper")
async def task_wrapper(
    task: str,
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    custom_prompt: str = Form(None)
):
    session_id = request.cookies.get("api_key_session")
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Load the dynamically generated module
        module_path = get_wrapper_path(task)
        if not module_path.exists():
            raise HTTPException(status_code=404, detail=f"Wrapper not found: {module_path}")

        spec = importlib.util.spec_from_file_location("task_module", module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)

        if not hasattr(task_module, "run_model"):
            raise HTTPException(status_code=500, detail="run_model function not in wrapper.")

        # Call the async run_model (which now POSTS to /api/analyze/)
        result = await task_module.run_model(temp_path, model, custom_prompt, session_id)
        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": "Wrapper execution failed",
            "exception": str(e)
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)