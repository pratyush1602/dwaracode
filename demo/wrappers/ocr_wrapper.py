"""
Auto-generated wrapper for task: Ocr
Calls /api/analyze/ API with user input and model.
"""
import requests
import os

def run_model(input_path: str, model: str, custom_prompt=None, session_id=None):
    url = "http://localhost:8000/api/analyze/"  # Ensure port matches your FastAPI app

    print(f"📂 Checking if file exists: {input_path}")
    if not os.path.exists(input_path):
        return {
            "error": f"Input file '{input_path}' not found"
        }

    with open(input_path, "rb") as f:
        files = {
            "file": f
        }
        payload = {
            "task_type": "Ocr",
            "model": model
        }
        if custom_prompt:
            payload["custom_prompt"] = custom_prompt
        if session_id:
            payload["session_id"] = session_id

        print("🚀 Sending request to /api/analyze")
        print(f"Payload: {payload}")
        try:
            response = requests.post(url, data=payload, files=files, timeout=30)
            response.raise_for_status()
            print("✅ Got response")
            return response.json()
        except requests.exceptions.RequestException as e:
            print("❌ Request failed:", str(e))
            return {
                "error": str(e),
                "details": getattr(e.response, 'text', None)
            }

