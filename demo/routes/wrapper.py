from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from typing import Optional
from ..utils import wrapper_exists
from ..utils import get_wrapper_path
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
    

@router.post("/wrappers/{task}_wrapper")
async def call_task_wrapper(
    task: str,
    model: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    if not wrapper_exists(task):
        raise HTTPException(status_code=404, detail="Wrapper not found for this task.")

    wrapper_path = get_wrapper_path(task)

    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(f"{task}_wrapper", wrapper_path)
        wrapper_module = importlib.util.module_from_spec(spec)
        sys.modules[f"{task}_wrapper"] = wrapper_module
        spec.loader.exec_module(wrapper_module)

        result = wrapper_module.call_analyze_api(
            file_path=temp_path,
            model=model,
            custom_prompt=custom_prompt,
            session_id=session_id
        )

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

