"""
Auto-generated wrapper for task: Accesslogs
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
        task_type="accesslogs",
        model=model,
        file=upload_file,
        custom_prompt=custom_prompt,
        session_id=session_id,
        api_key_session=None
    )
