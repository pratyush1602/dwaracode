import requests

def call_analyze_api(
    file_path: str,
    model: str = None,
    custom_prompt: str = None,
    session_id: str = None,
    api_url: str = "http://localhost:8000/api/analyze/"
):
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        data = {
            "task_type": "ocr"
        }
        if model:
            data["model"] = model
        if custom_prompt:
            data["custom_prompt"] = custom_prompt
        if session_id:
            data["session_id"] = session_id

        response = requests.post(api_url, data=data, files=files)
        
        if response.status_code == 200:
            print("Response:")
            print(response.json())
            return response.json()
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None

# Example usage
if __name__ == "__main__":
    call_analyze_api("example.log", model="your_model_id", custom_prompt="Explain the log", session_id="abc123")
