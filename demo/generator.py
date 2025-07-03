from .utils import get_wrapper_path


def create_wrapper_code(task: str) -> str:
    return f'''import requests

def call_analyze_api(
    file_path: str,
    model: str = None,
    custom_prompt: str = None,
    session_id: str = None,
    api_url: str = "http://localhost:8005/api/analyze/"
):
    with open(file_path, "rb") as f:
        files = {{"file": (file_path, f)}}
        data = {{
            "task_type": "{task}"
        }}
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
'''



def generate_wrapper(task: str) -> str:
    path = get_wrapper_path(task)
    code = create_wrapper_code(task)
    with open(path, "w") as f:
        f.write(code)
    return str(path)