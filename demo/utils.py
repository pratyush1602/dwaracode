from pathlib import Path

WRAPPER_DIR = Path(__file__).parent / "wrappers"
WRAPPER_DIR.mkdir(exist_ok=True)

def get_wrapper_path(task: str) -> Path:
    filename = f"{task}_wrapper.py".replace("-","_")
    return WRAPPER_DIR / filename

def wrapper_exists(task: str)-> Path:
    return get_wrapper_path(task).exists()