from pydantic import BaseModel
# from typing import Optional
# from fastapi import UploadFile

# class LogAnalysisRequest(BaseModel):
#     log_type: str  # e.g., "errorlogs", "systemlogs"
#     model: Optional[str] = None  # Optional model override
#     file: UploadFile  # Log file to be analyzed

class LogAnalysisResponse(BaseModel):
    response: str  # The generated response from the model
    used_model: str  # The model used for generating the response
    log_type: str  # Type of log analyzed