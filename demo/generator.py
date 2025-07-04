from demo.utils import get_wrapper_path
from pathlib import Path

def create_wrapper_code(task: str) -> str:
    task_cap = task.capitalize()

    return f'''"""
Auto-generated wrapper for task: {task_cap}
Calls /api/analyze/ API with user input and model.
"""
import requests
import os

def run_model(input_path: str, model: str, custom_prompt=None, session_id=None):
    url = "http://localhost:8000/api/analyze/"

    if not os.path.exists(input_path):
        return {{
            "error": f"Input file '{{input_path}}' not found"
        }}

    with open(input_path, "rb") as f:
        files = {{
            "file": f
        }}
        payload = {{
            "task_type": "{task}",
            "model": model
        }}

        if custom_prompt:
            payload["custom_prompt"] = custom_prompt
        if session_id:
            payload["session_id"] = session_id

        try:
            response = requests.post(url, data=payload, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {{
                "error": str(e),
                "details": getattr(e.response, 'text', None)
            }}
'''

def generate_wrapper(task: str) -> str:
    path = get_wrapper_path(task)  # This gives demo/wrappers/<Task>_wrapper.py
    code = create_wrapper_code(task)

    wrapper_dir = Path(path).parent
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"✅ Wrapper generated at {path}")
    return str(path)
