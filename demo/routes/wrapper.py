from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from typing import Optional
from demo.utils import wrapper_exists, get_wrapper_path
from demo.generator import generate_wrapper
from pathlib import Path
import importlib.util
import os
import traceback

router = APIRouter()

# ========================
# ✅ 1. Generate Wrapper
# ========================
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
        path = generate_wrapper(task_type)
        return {
            "message": f"Wrapper generated at {path}",
            "task_type": task_type,
            "used_model": model,
            "response": "Sample response",  # Replace later with real result
            "saved_files": {
                "request": {"txt": "path/to/input.txt"},
                "response": {"json": "path/to/output.json"}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wrapper generation failed: {str(e)}")


# ========================
# ✅ 2. Call Wrapper
# ========================
@router.post("/wrappers/{task}_wrapper")
async def task_wrapper(
    task: str,
    file: UploadFile = File(...),
    model: str = Form(...),
    custom_prompt: str = Form(None),
    session_id: str = Form(None)
):
    temp_path = f"temp_{file.filename}"

    try:
        # Save uploaded file to a temporary location
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Load dynamically generated wrapper from correct path
        module_path = f"demo/wrappers/{task}_wrapper.py"
        if not os.path.exists(module_path):
            raise HTTPException(status_code=404, detail=f"Wrapper file not found at {module_path}")

        spec = importlib.util.spec_from_file_location("task_module", module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)

        if not hasattr(task_module, "run_model"):
            raise HTTPException(status_code=500, detail=f"No run_model() function in wrapper {task}_wrapper.py")

        # Execute the wrapper's run_model function
        result = task_module.run_model(temp_path, model, custom_prompt, session_id)
        return result

    except Exception as e:
        print("=== Wrapper Execution Error ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in wrapper execution: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
