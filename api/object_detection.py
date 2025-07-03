from fastapi import APIRouter, UploadFile, File, WebSocket, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
from PIL import Image
import base64
import json
import os
import google.generativeai as genai
from typing import Optional, Dict
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
# from fastapi_cache.backends.redis import RedisBackend
# from redis import asyncio as aioredis
from collections import deque
import asyncio
# from fastapi_utils.tasks import repeast_every
import uuid
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import fractions
from fastapi import WebSocket, WebSocketDisconnect

object_detection_router = APIRouter()

# Initialize YOLO model with correct paths
net = cv2.dnn.readNet(
    # "models/yolov3.weights",
    "models/yolov3.cfg"
)
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Configure Google Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your_api_key_here")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# File upload endpoint
@object_detection_router.post("/upload")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        height, width = image_np.shape[:2]
        
        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Prepare results
        results = []
        for i in range(len(boxes)):
            if i in indexes:
                results.append({
                    "class": classes[class_ids[i]],
                    "confidence": round(confidences[i] * 100, 2),
                    "box": boxes[i]
                })
        
        return JSONResponse(content={"objects": results})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# WebSocket endpoint
@object_detection_router.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive image data
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process image
                height, width = img.shape[:2]
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                
                # Process detections
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                # Apply non-maximum suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                # Prepare results
                results = []
                for i in range(len(boxes)):
                    if i in indexes:
                        results.append({
                            "class": classes[class_ids[i]],
                            "confidence": round(confidences[i] * 100, 2),
                            "box": boxes[i]
                        })
                
                # Send results back to client
                await websocket.send_text(json.dumps(results))
                
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

class RequestQueue:
    def __init__(self):
        self.queue = deque()
        self.processing = False

    async def add_request(self, image, question):
        future = asyncio.Future()
        self.queue.append((image, question, future))
        if not self.processing:
            asyncio.create_task(self.process_queue())
        return await future

    async def process_queue(self):
        self.processing = True
        while self.queue:
            image, question, future = self.queue.popleft()
            try:
                result = await self._process_request(image, question)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        self.processing = False

request_queue = RequestQueue()

class ProgressTracker:
    def __init__(self):
        self.tasks = {}

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None
        }
        return task_id

    def update_progress(self, task_id: str, progress: int):
        if task_id in self.tasks:
            self.tasks[task_id]["progress"] = progress

progress_tracker = ProgressTracker()

@object_detection_router.post("/ask")
@cache(expire=3600)  # Cache for 1 hour
async def answer_question(
    file: UploadFile = File(..., description="Image file under 5MB"),
    question: str = Form(...),
    max_file_size: int = 5 * 1024 * 1024  # 5MB limit
):
    task_id = progress_tracker.create_task()
    
    try:
        # Check cache first
        cached_response = await FastAPICache.get(f"{hash(await file.read())}-{question}")
        if cached_response:
            return JSONResponse(content=cached_response)

        # Validate file size before processing
        file_size = 0
        contents = bytearray()
        
        # Read file in chunks to prevent memory issues
        async for chunk in file.iter_chunks(8192):  # 8KB chunks
            file_size += len(chunk)
            if file_size > max_file_size:
                raise HTTPException(status_code=413, detail="File too large")
            contents.extend(chunk)
            
        # Process image with progress updates
        progress_tracker.update_progress(task_id, 25)
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        progress_tracker.update_progress(task_id, 50)
        # Optimize image size if needed
        if image.size[0] > 1280 or image.size[1] > 1280:
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        
        progress_tracker.update_progress(task_id, 75)
        # Prepare a more general prompt for object identification
        prompt = f"""
        Please analyze this image and answer the following question:
        
        Question: {question}
        
        Focus on identifying objects, people, animals, scenes, activities, and other visual elements present in the image.
        Describe their appearance, position, and relationships if relevant to the question.
        
        If asked to identify what's in the image, list the main objects and elements visible.
        If asked about specific objects, provide detailed information about those objects.
        If asked about actions or activities, describe what is happening in the image.
        
        Provide a concise and accurate answer based on what you can see in the image.
        """
        
        # Convert PIL Image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create the content parts for Gemini
        contents = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')}}
        ]
        
        # Send the image and prompt to Gemini
        response = gemini_model.generate_content(contents)
        
        # Extract the answer from Gemini's response
        answer = response.text.strip()
        
        # Determine confidence based on the language used in the response
        confidence = 0.9  # Default high confidence for Gemini
        
        # Lower confidence if the answer contains uncertainty markers
        uncertainty_phrases = ["appears to be", "might be", "could be", "possibly", "I'm not sure", "uncertain"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence = 0.7
        
        # Return the answer
        result = {
            "answer": answer,
            "confidence": confidence
        }
        
        progress_tracker.update_progress(task_id, 100)
        return JSONResponse(content={
            "task_id": task_id,
            "status": "completed",
            "result": result
        })
        
    except Exception as e:
        progress_tracker.tasks[task_id]["status"] = "failed"
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in answer_question: {error_details}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Add a new endpoint specifically for document identification
@object_detection_router.post("/identify_objects")
async def identify_objects(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read and process the uploaded image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Prepare a general object identification prompt
        prompt = """
        Please identify and list the main objects visible in this image.
        
        For each significant object, provide:
        1. The name of the object
        2. A brief description of its appearance
        3. Its approximate location in the image (if relevant)
        
        Focus on the most prominent and important objects first.
        If there are people in the image, mention them but respect privacy by not attempting to identify specific individuals.
        
        Format your response as a simple list of objects.
        """
        
        # Convert PIL Image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create the content parts for Gemini
        contents = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')}}
        ]
        
        try:
            # Send the image and prompt to Gemini
            response = gemini_model.generate_content(contents)
            
            # Extract the objects list from Gemini's response
            objects_description = response.text.strip()
            
            # Return the objects description
            return JSONResponse(content={
                "objects": objects_description,
                "confidence": 0.9
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image with Gemini: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in identify_objects: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

# Store active KYC sessions
kyc_sessions: Dict[str, Dict] = {}

# Directory to store conversation logs
os.makedirs("data/conversations", exist_ok=True)

async def analyze_document_with_gemini(image_data: str) -> dict:
    try:
        print("Starting Gemini document analysis...")
        
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Prepare prompt for document analysis
        prompt = """
        Analyze this image for ID document verification. First, check if this is a valid ID document.
        If it is a valid ID, extract all visible information such as:
        - Full name
        - Date of birth
        - ID/Document number
        - Expiry date
        - Document type
        - Other visible details

        Respond with a JSON object containing:
        {
            "is_valid_document": boolean,
            "document_type": string,
            "quality_score": number (1-10),
            "extracted_details": {
                "full_name": string,
                "date_of_birth": string,
                "document_number": string,
                "expiry_date": string,
                "other_details": object
            },
            "data_visibility": {
                "name_visible": boolean,
                "dob_visible": boolean,
                "number_visible": boolean,
                "expiry_visible": boolean,
                "photo_visible": boolean
            },
            "security_features": {
                "hologram_detected": boolean,
                "photo_matches_video": boolean,
                "appears_genuine": boolean
            },
            "issues_found": string[],
            "recommendations": string[]
        }

        Important privacy guidelines:
        1. If the document is valid but blurry/unclear, mention it in issues but don't guess at unreadable information
        2. If any security feature seems suspicious, note it in issues
        3. For any unreadable fields, use null instead of guessing
        4. If the document appears tampered or fake, set is_valid_document to false

        Analyze the image thoroughly and provide accurate details while following privacy and security guidelines.
        """

        # Create Gemini request
        contents = [
            {"text": prompt},
            {"inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode()
            }}
        ]
        
        print("Sending request to Gemini...")
        response = gemini_model.generate_content(contents)
        print(f"Received response from Gemini: {response}")
        
        try:
            # Extract JSON from markdown code block
            response_text = response.text
            if "```json" in response_text:
                # Extract content between ```json and ```
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                # If no markdown, use the raw text
                json_str = response_text.strip()
            
            print("Extracted JSON string:", json_str)
            result = json.loads(json_str)
            print("Parsed analysis result:", result)
            return result
            
        except json.JSONDecodeError as e:
            print("Error parsing Gemini response:", e)
            print("Raw response text:", response.text)
            return {
                "is_valid_document": False,
                "document_type": "unknown",
                "quality_score": 0,
                "extracted_details": {},
                "data_visibility": {},
                "security_features": {},
                "issues_found": ["Could not analyze document properly", f"Parse error: {str(e)}"],
                "recommendations": ["Please try capturing again"]
            }

    except Exception as e:
        print(f"Error in document analysis: {e}")
        return {
            "is_valid_document": False,
            "document_type": "unknown",
            "quality_score": 0,
            "extracted_details": {},
            "data_visibility": {},
            "security_features": {},
            "issues_found": [str(e)],
            "recommendations": ["Technical error occurred, please try again"]
        }

async def save_conversation_message(session_id: str, sender: str, message: str):
    """Save a conversation message to the session and persist to disk"""
    timestamp = int(asyncio.get_event_loop().time())
    
    # Initialize conversation history if it doesn't exist
    if "conversation" not in kyc_sessions[session_id]:
        kyc_sessions[session_id]["conversation"] = []
    
    # Add message to in-memory history
    message_record = {
        "timestamp": timestamp,
        "sender": sender,
        "message": message
    }
    kyc_sessions[session_id]["conversation"].append(message_record)
    
    # Save to disk
    conversation_file = f"data/conversations/{session_id}.json"
    try:
        # If file exists, read it first
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)
        else:
            conversation_data = {"session_id": session_id, "messages": []}
        
        # Append new message
        conversation_data["messages"].append(message_record)
        
        # Write back to file
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
    except Exception as e:
        print(f"Error saving conversation: {e}")

@object_detection_router.websocket("/kyc/session/{session_id}")
async def kyc_session_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in kyc_sessions:
        kyc_sessions[session_id] = {
            "admin_ws": None,
            "customer_ws": None,
            "conversation": []
        }
    
    kyc_sessions[session_id]["customer_ws"] = websocket
    
    # Notify admin that customer has connected
    if kyc_sessions[session_id]["admin_ws"]:
        await kyc_sessions[session_id]["admin_ws"].send_json({
            "type": "customer_connected"
        })
    
    try:
        while True:
            message = await websocket.receive_json()
            
            # Handle WebRTC signaling messages
            if message["type"] in ["offer", "ice_candidate"] and kyc_sessions[session_id]["admin_ws"]:
                await kyc_sessions[session_id]["admin_ws"].send_json(message)
            
            # Handle chat messages from customer
            elif message["type"] == "chat_message" and "message" in message:
                # Save the message to conversation history
                await save_conversation_message(session_id, "Customer", message["message"])
                
                # Forward the message to admin
                if kyc_sessions[session_id]["admin_ws"]:
                    await kyc_sessions[session_id]["admin_ws"].send_json({
                        "type": "chat_message",
                        "sender": "Customer",
                        "message": message["message"]
                    })
            
    except WebSocketDisconnect:
        print(f"Customer disconnected from session {session_id}")
    finally:
        if session_id in kyc_sessions:
            kyc_sessions[session_id]["customer_ws"] = None
            if not kyc_sessions[session_id]["admin_ws"]:
                del kyc_sessions[session_id]
        await websocket.close()

@object_detection_router.websocket("/kyc/admin/{session_id}")
async def kyc_admin_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in kyc_sessions:
        kyc_sessions[session_id] = {
            "admin_ws": websocket,
            "customer_ws": None,
            "conversation": []
        }
    else:
        kyc_sessions[session_id]["admin_ws"] = websocket
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if message["type"] == "analyze_document":
                print("Admin requested document analysis")
                # Only call Gemini when admin captures
                result = await analyze_document_with_gemini(message["data"])
                
                # Send analysis results back to admin
                await websocket.send_json({
                    "type": "document_analysis",
                    "result": result
                })
                
                # Notify customer of the capture result
                if kyc_sessions[session_id]["customer_ws"]:
                    await kyc_sessions[session_id]["customer_ws"].send_json({
                        "type": "document_captured",
                        "result": result
                    })
            
            # Handle chat messages from admin
            elif message["type"] == "chat_message" and "message" in message:
                # Save the message to conversation history
                await save_conversation_message(session_id, "Admin", message["message"])
                
                # Forward the message to customer
                if kyc_sessions[session_id]["customer_ws"]:
                    await kyc_sessions[session_id]["customer_ws"].send_json({
                        "type": "chat_message",
                        "sender": "Admin",
                        "message": message["message"]
                    })
            
            # Handle instructional messages (pre-existing feature)
            elif message["type"] == "instruction" and "message" in message:
                # Save the instruction as a message in the conversation
                await save_conversation_message(session_id, "Admin", message["message"])
                
                # Forward to customer
                if kyc_sessions[session_id]["customer_ws"]:
                    await kyc_sessions[session_id]["customer_ws"].send_json(message)
            
            # Handle other message types
            elif message["type"] in ["answer", "ice_candidate"]:
                if kyc_sessions[session_id]["customer_ws"]:
                    await kyc_sessions[session_id]["customer_ws"].send_json(message)
            
    except WebSocketDisconnect:
        print(f"Admin disconnected from session {session_id}")
    finally:
        if session_id in kyc_sessions:
            kyc_sessions[session_id]["admin_ws"] = None
            if not kyc_sessions[session_id]["customer_ws"]:
                del kyc_sessions[session_id]
        await websocket.close()

# Endpoint to retrieve conversation history
@object_detection_router.get("/kyc/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    # Check if conversation exists in memory
    if session_id in kyc_sessions and "conversation" in kyc_sessions[session_id]:
        return {"session_id": session_id, "messages": kyc_sessions[session_id]["conversation"]}
    
    # Check if conversation exists on disk
    conversation_file = f"data/conversations/{session_id}.json"
    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading conversation: {str(e)}")
    
    # No conversation found
    raise HTTPException(status_code=404, detail="Conversation not found")

# Add a helper function to debug sessions
@object_detection_router.get("/kyc/debug/sessions")
async def debug_sessions():
    return {
        session_id: {
            "has_admin": bool(session["admin_ws"]),
            "has_customer": bool(session["customer_ws"])
        }
        for session_id, session in kyc_sessions.items()
    } 