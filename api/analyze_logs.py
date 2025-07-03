# from fastapi import APIRouter, UploadFile, File, Form
# import json
# from pathlib import Path

# from models.analyze_logs import LogAnalysisResponse
# from services.provider_client import generate_response
# from services.helper_functions import load_json

# analyze_logs_router = APIRouter()

# @analyze_logs_router.post("/")
# async def analyze_logs(
#     log_type: str = Form(None),  # e.g., "errorlogs", "systemlogs", etc.
#     model: str = Form(None),    # Optional model override
#     file: UploadFile = File(...)
# ):
#     # Read config file
#     config = load_json("json/config.json")
        
#     # Get the configuration for this log type
#     log_config = config.get(log_type, config["general"])[0] # Using first config entry for the log type
    
#     # Use provided model or fall back to config-specified model
#     selected_model = model or log_config["model"]
    
#     # Read and parse the uploaded file
#     file_content = await file.read()
#     json_data = json.loads(file_content)
#     error_message = json_data.get("error", "No error message found")
    
#     # Use the prompt from config
#     prompt = f"{log_config['prompt']}\n\n{error_message}"
#     # Generate response using the selected model
#     response = await generate_response(prompt, model=selected_model)
#     print(response)
#     return LogAnalysisResponse(
#         response= response,
#         used_model= selected_model,
#         log_type= log_type
#     )