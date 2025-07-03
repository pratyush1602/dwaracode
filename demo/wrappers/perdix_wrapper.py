"""
Auto-generated wrapper for task: perdix
Calls /analyze API with user input and model.
"""

import requests

def run_model(input_data, model):
    url = "http://localhost:8000/analyze"
    payload = {
        "taskType": "perdix",
        "modelName": model,
        "inputData": input_data
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "details": getattr(e.response, 'text', None)
        }

# Optional test run
if __name__ == "__main__":
    sample_input = "Sample input text or data"
    sample_model = "gpt-4"
    result = run_model(sample_input, sample_model)
    print("Wrapper Output:")
    print(result)
