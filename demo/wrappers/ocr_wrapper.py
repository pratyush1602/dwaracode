"""
Auto-generated wrapper for task: Ocr
Calls /api/analyze/ using an HTTP POST request.
"""
print("github")
import httpx
from io import BytesIO
from starlette.datastructures import UploadFile

class SimpleUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes):
        super().__init__(filename=filename, file=BytesIO(content))

async def run_model(file_path: str, model: str, custom_prompt=None, session_id=None):
    url = "http://localhost:8000/api/analyze/"

    # Read file bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Build form data
    form_data = {
        "task_type": "ocr",
        "model": model
    }
    if custom_prompt:
        form_data["custom_prompt"] = custom_prompt

    # Files payload
    files = {
        "file": (file_path.split("/")[-1], file_bytes, "application/octet-stream")
    }

    # Add cookie if session_id is available
    cookies = {"api_key_session": session_id} if session_id else None

    async with httpx.AsyncClient(cookies=cookies) as client:
        response = await client.post(
            url,
            data=form_data,
            files=files
        )

    if response.status_code != 200:
        return {
            "status": "error",
            "code": response.status_code,
            "error": response.text
        }

    return response.json()
