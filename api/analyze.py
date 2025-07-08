from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Cookie
import json
from pathlib import Path
import io
from PIL import Image
import pytesseract
import mimetypes
import re
from datetime import datetime
import os
import shutil

import logging

logger = logging.getLogger(__name__)

# You'll need to install these packages:
# pip install pytesseract pillow python-multipart PyPDF2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    from PyPDF2 import PdfReader
except ImportError:
    # For compatibility with different PyPDF2 versions
    try:
        from PyPDF2 import PdfFileReader as PdfReader
    except ImportError:
        PdfReader = None

from models.analyze import LogAnalysisResponse
from services.provider_client import generate_response, analyze_image_with_vision_model, analyze_pdf_with_vision_model
from services.helper_functions import load_json

analyze_logs_router = APIRouter()

def save_request_and_response(file: UploadFile, response_data, task_type: str):
    """Save both request file and response data in organized folders"""
    # Create base directories
    base_dir = Path("data/ocr")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure: data/ocr/YYYYMMDD_HHMMSS_tasktype/
    session_dir = base_dir / f"{timestamp}_{task_type}"
    request_dir = session_dir / "request"
    response_dir = session_dir / "response"
    
    # Create all directories
    for dir_path in [base_dir, session_dir, request_dir, response_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {
        "request": {},
        "response": {}
    }
    
    # Save request file
    try:
        # Get file extension from original filename
        file_ext = os.path.splitext(file.filename)[1]
        request_filename = f"original{file_ext}"
        request_path = request_dir / request_filename
        
        # Save the original file
        with open(request_path, "wb") as f:
            # Reset file pointer to beginning
            file.file.seek(0)
            shutil.copyfileobj(file.file, f)
            
        saved_files["request"]["original"] = str(request_path)
        
        # If it's an image, save the OCR text separately
        if file.content_type and "image" in file.content_type:
            ocr_text_path = request_dir / "ocr_text.txt"
            image = Image.open(request_path)
            ocr_text = pytesseract.image_to_string(image)
            
            with open(ocr_text_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            
            saved_files["request"]["ocr_text"] = str(ocr_text_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving request file: {str(e)}")
    
    # Save response in different formats
    try:
        # Save as JSON
        json_path = response_dir / "response.json"
        json_content = {
            "model_used": response_data.used_model,
            "task_type": response_data.log_type,
            "response": response_data.response,
            "timestamp": datetime.now().isoformat(),
            "original_filename": file.filename
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=2)
        saved_files["response"]["json"] = str(json_path)
        
        # Save as TXT
        txt_path = response_dir / "response.txt"
        txt_content = f"""Analysis Response
Original File: {file.filename}
Model Used: {response_data.used_model}
Task Type: {response_data.log_type}
Timestamp: {datetime.now().isoformat()}

=== Response ===

{response_data.response}"""
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        saved_files["response"]["txt"] = str(txt_path)
        
        # Save as HTML
        html_path = response_dir / "response.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Response</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        pre {{ white-space: pre-wrap; background: #f5f5f5; padding: 15px; }}
    </style>
</head>
<body>
    <h2>Analysis Response</h2>
    <p><strong>Original File:</strong> {file.filename}</p>
    <p><strong>Model Used:</strong> {response_data.used_model}</p>
    <p><strong>Task Type:</strong> {response_data.log_type}</p>
    <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
    <pre>{response_data.response}</pre>
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        saved_files["response"]["html"] = str(html_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving response files: {str(e)}")
    
    return saved_files

@analyze_logs_router.post("/")
async def analyze(
    task_type: str = Form(...),  # e.g., "errorlogs", "systemlogs", etc.
    model: str = Form(None),    # Optional model override
    file: UploadFile = File(...),
    custom_prompt: str = Form(None),  # Add custom_prompt parameter
    session_id: str = Form(None),  # For explicit session IDs from form data
    api_key_session: str = Cookie(None)  # Get session from cookies automatically
):
    # Use the cookie session if available and no explicit session ID provided
    effective_session_id = session_id or api_key_session
    
    # Read config file
    config = load_json("json/config.json")
    
    # Log session usage
    if effective_session_id:
        print(f"Using session ID from {'cookie' if api_key_session else 'form'}: {effective_session_id[:8]}...")
    
    # Validate log type exists in config
    if task_type not in config:
        return {"error": f"Invalid log type: {task_type}"}
    
    # Get the configuration for this log type
    log_config = config[task_type][0]  # Using first config entry for the log type
    
    # Use provided model or fall back to config-specified model
    selected_model = model or log_config["model"]
    
    # Determine file type and extract content accordingly
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    file_content = await file.read()
    
    # Process based on file type
    error_message = ""
    use_vision_model = False
    
    if content_type and "image" in content_type:
        # Handle image files with vision model
        use_vision_model = True
        try:
            # Create a temporary file to pass to the vision model
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            # Use the analyze_image_with_vision_model function with session ID
            response = await analyze_image_with_vision_model(
                temp_file_path, 
                custom_prompt or log_config['prompt'], 
                selected_model,
                session_id=effective_session_id  # Pass the effective session ID
            )
            
            # Clean up the temporary file
            os.remove(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    elif content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
        # Handle PDF files with dedicated PDF analysis function
        use_vision_model = True
        try:
            # Create a temporary file to pass to the PDF vision model
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            # Use the dedicated PDF analysis function with session ID
            response = await analyze_pdf_with_vision_model(
                temp_file_path, 
                custom_prompt or log_config['prompt'], 
                selected_model,
                session_id=effective_session_id  # Pass the effective session ID
            )
            
            # Clean up the temporary file
            os.remove(temp_file_path)
        except Exception as e:
            # Fallback to traditional PDF text extraction if vision model fails
            use_vision_model = False
            logger.warning(f"Vision model PDF analysis failed: {str(e)}. Falling back to text extraction.")
            
            if not PdfReader:
                raise HTTPException(status_code=400, detail="PDF processing is not available. Install PyPDF2.")
            
            try:
                pdf = PdfReader(io.BytesIO(file_content))
                error_message = ""
                for page_num in range(len(pdf.pages)):
                    page = pdf.pages[page_num]
                    error_message += page.extract_text() + "\n"
            except Exception as pdf_error:
                raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(pdf_error)}")
    
    elif content_type == "application/json" or file.filename.lower().endswith('.json'):
        # Handle JSON files
        try:
            json_data = json.loads(file_content)
            error_message = json_data.get("error", json.dumps(json_data, indent=2))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    
    else:
        # Handle as plain text
        try:
            error_message = file_content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Unsupported file type or encoding")
    
    # Clean up the extracted text if needed
    error_message = error_message.strip()
    if not error_message:
        error_message = "No content could be extracted from the file."
    
    # Use custom prompt if provided, otherwise use the prompt from config
    prompt_template = custom_prompt if custom_prompt else log_config['prompt']
    prompt = f"{prompt_template}\n\n{error_message}"
    
    # Generate response using the selected model
    if not use_vision_model:
        try:
            response = await generate_response(prompt, selected_model, session_id=effective_session_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating response: {str(e)}")
    
    # Create response object
    analysis_response = LogAnalysisResponse(
        response=response,
        used_model=selected_model,
        log_type=task_type
    )
    
    # Save both request and response files
    saved_files = save_request_and_response(file, analysis_response, task_type)
    
    # Also save the custom prompt if it was used
    if custom_prompt:
        try:
            # Save the custom prompt to the request directory
            request_dir = Path(saved_files["request"]["original"]).parent
            prompt_path = request_dir / "custom_prompt.txt"
            
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(custom_prompt)
            
            saved_files["request"]["custom_prompt"] = str(prompt_path)
        except Exception as e:
            print(f"Error saving custom prompt: {e}")
    
    return {
        **analysis_response.dict(),
        "saved_files": saved_files
    }

@analyze_logs_router.get("/history")
async def get_analysis_history():
    """Get list of all previously analyzed files"""
    base_dir = Path("data/ocr")
    
    # Return empty list if directory doesn't exist
    if not base_dir.exists():
        return {"sessions": [], "status": "success"}
    
    sessions = []
    
    try:
        # List all session directories
        for session_dir in sorted(base_dir.iterdir(), reverse=True):  # Latest first
            if session_dir.is_dir():
                try:
                    # Parse session info from directory name
                    dir_name = session_dir.name
                    
                    # Extract timestamp and task_type
                    timestamp_str = dir_name[:14]  # Get first 14 characters (YYYYMMDD_HHMMSS)
                    task_type = dir_name[15:]      # Get everything after the underscore
                    
                    # Parse timestamp with correct format
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Get original file info
                    request_dir = session_dir / "request"
                    response_dir = session_dir / "response"
                    
                    files = {
                        "request": [],
                        "response": []
                    }
                    
                    # List request files
                    if request_dir.exists():
                        files["request"] = [
                            str(f.relative_to(base_dir)) 
                            for f in request_dir.glob("*") 
                            if f.is_file()
                        ]
                    
                    # List response files
                    if response_dir.exists():
                        files["response"] = [
                            str(f.relative_to(base_dir)) 
                            for f in response_dir.glob("*") 
                            if f.is_file()
                        ]
                    
                    # Get original filename if available
                    original_file = None
                    if files["request"]:
                        original_file = Path(files["request"][0]).name
                    
                    sessions.append({
                        "id": session_dir.name,
                        "timestamp": timestamp.isoformat(),
                        "task_type": task_type,
                        "original_file": original_file,
                        "files": files,
                        "path": str(session_dir.relative_to(base_dir))
                    })
                except Exception as e:
                    print(f"Error processing session {session_dir}: {e}")
                    continue
                    
        return {
            "status": "success",
            "sessions": sessions
        }
    except Exception as e:
        print(f"Error reading history: {e}")
        return {
            "status": "error",
            "message": str(e),
            "sessions": []
        }