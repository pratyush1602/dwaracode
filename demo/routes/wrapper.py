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