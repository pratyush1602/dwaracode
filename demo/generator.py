from demo.utils import get_wrapper_path
from pathlib import Path

def create_wrapper_code(task: str) -> str:
    return f'''"""
Auto-generated wrapper for task: {task}
Calls /api/analyze/ using an HTTP POST request.
"""
import httpx
from io import BytesIO
from starlette.datastructures import UploadFile

class SimpleUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes):
        super().__init__(filename=filename, file=BytesIO(content))

async def run_model(file_path: str, model: str, custom_prompt=None):
    analyze_url = "http://localhost:8000/api/analyze/"
    token_url = "http://localhost:8000/unlimited-token/internal"

    try:
        # Step 1: Fetch the token
        async with httpx.AsyncClient() as client:
            token_response = await client.get(token_url)
            if token_response.status_code != 200:
                return {{
                    "status": "error",
                    "stage": "get-token",
                    "code": token_response.status_code,
                    "body": token_response.text
                }}
            token_data = token_response.json()
            access_token = token_data.get("access_token")

        if not access_token:
            return {{
                "status": "error",
                "stage": "token-missing",
                "error": "Access token not found in response"
            }}

        # Step 2: Read file bytes
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Step 3: Prepare request
        form_data = {{
            "task_type": "{task}",
            "model": model
        }}
        if custom_prompt:
            form_data["custom_prompt"] = custom_prompt

        files = {{
            "file": (file_path.split("/")[-1], file_bytes, "application/octet-stream")
        }}

        headers = {{
            "Authorization": f"Bearer {{access_token}}"
        }}

        # Step 4: Send analyze request
        async with httpx.AsyncClient() as client:
            response = await client.post(analyze_url, data=form_data, files=files, headers=headers)

        if response.status_code != 200:
            return {{
                "status": "error",
                "stage": "analyze",
                "code": response.status_code,
                "body": response.text
            }}

        return response.json()

    except Exception as e:
        import traceback
        return {{
            "status": "error",
            "message": "Wrapper execution failed",
            "exception": str(e),
            "trace": traceback.format_exc()
        }}
'''

def generate_wrapper(task: str) -> str:
    path = get_wrapper_path(task)
    code = create_wrapper_code(task)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Wrapper generated at {path}")
    return str(path)
