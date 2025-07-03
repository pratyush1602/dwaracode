from utils import get_wrapper_path


def create_wrapper_code(task: str) -> str:
    return f"# This is a placeholder wrapper for task: {task}\n# Just for testing and demonstration."


def generate_wrapper(task: str) -> str:
    path = get_wrapper_path(task)
    code = create_wrapper_code(task)
    with open(path, "w") as f:
        f.write(code)
    return str(path)