from utils import wrapper_exists, get_wrapper_path
from generator import generate_wrapper

task = "adhar"

# Check if the wrapper exists
if wrapper_exists(task):
    print(f"Wrapper already exists for task: {task}")
    print("Path:", get_wrapper_path(task))
else:
    print(f" Wrapper does NOT exist for task: {task}")
    print(" Expected path would be:", get_wrapper_path(task))
    
    print("Generating wrapper now...")
    path = generate_wrapper(task)
    print(f"Wrapper generated at: {path}")

    # Confirm it now exists
    if wrapper_exists(task):
        print(f"Confirmed: Wrapper now exists at {get_wrapper_path(task)}")
    else:
        print("Something went wrong. Wrapper still doesn't exist.")
