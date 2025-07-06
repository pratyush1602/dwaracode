from demo.utils import get_wrapper_path
from pathlib import Path

# generator.py
def create_wrapper_code(task: str) -> str:
    task_cap = task.capitalize()
    return f'''"""
Auto-generated wrapper for task: {task_cap}
Directly calls analyze(...) function.
"""
from api.analyze import analyze
from starlette.datastructures import UploadFile
from io import BytesIO

class SimpleUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes):
        super().__init__(filename=filename, file=BytesIO(content))

async def run_model(file_path: str, model: str, custom_prompt=None, session_id=None):
    with open(file_path, "rb") as f:
        content = f.read()
        filename = file_path.split("/")[-1]
        upload_file = SimpleUploadFile(filename, content)

    return await analyze(
        task_type="{task}",
        model=model,
        file=upload_file,
        custom_prompt=custom_prompt,
        session_id=session_id,
        api_key_session=None
    )
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
