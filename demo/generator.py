from .utils import get_wrapper_path
import base64

def create_wrapper_code(
    task: str,
    model: str,
    custom_prompt: str,
    filename: str,
    file_content: bytes,
    token: str,
    api_url: str = "http://localhost:8005/api/analyze/"
) -> str:
    # Encode the file content to base64
    b64_content = base64.b64encode(file_content).decode("utf-8")

    return f'''import requests
import base64
from io import BytesIO

# === Configuration ===
API_URL = "{api_url}"
TOKEN = "{token}"

# === Form Data ===
task_type = "{task}"
model = "{model}"
custom_prompt = """{custom_prompt}"""

# === File to Send ===
filename = "{filename}"
b64_content = \"\"\"{b64_content}\"\"\"
file_bytes = base64.b64decode(b64_content)
file_content = BytesIO(file_bytes)

# === Make the request ===
files = {{'file': (filename, file_content)}}
data = {{
    'task_type': task_type,
    'model': model,
    'custom_prompt': custom_prompt
}}
headers = {{
    'Authorization': f'Bearer {{TOKEN}}'
}}

response = requests.post(API_URL, data=data, files=files, headers=headers)

# === Handle Response ===
if response.ok:
    result = response.json()
    print("Analysis successful!")
    print("Used Model:", result.get("used_model"))
    print("Task Type:", result.get("log_type"))
    print("Response:\\n", result.get("response"))
else:
    print("Failed:", response.status_code, response.text)
'''


def generate_wrapper(
    task: str,
    model: str,
    custom_prompt: str,
    filename: str,
    file_bytes: bytes,
    token: str,
    api_url: str
) -> str:
    path = get_wrapper_path(task)
    code = create_wrapper_code(
        task=task,
        model=model,
        custom_prompt=custom_prompt,
        filename=filename,
        file_content=file_bytes,
        token=token,
        api_url=api_url
    )
    with open(path, "w") as f:
        f.write(code)
    return str(path)
