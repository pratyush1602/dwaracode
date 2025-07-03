# from fastapi import APIRouter, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import io
# import os
# import json
# import numpy as np

# document_classifier_router = APIRouter()

# # Define the document types
# DOCUMENT_TYPES = {
#     0: "Aadhaar Card",
#     1: "PAN Card",
#     2: "TAN Card",
#     3: "Driving License",
#     4: "Voter ID",
#     5: "Other Document"
# }

# class DocumentClassifier(nn.Module):
#     def __init__(self, num_classes=6):
#         super(DocumentClassifier, self).__init__()
#         # Use a pre-trained ResNet model
#         self.model = models.resnet50(pretrained=True)
        
#         # Replace the final fully connected layer
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         return self.model(x)

# # Initialize the model
# model_path = "models/document_classifier.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create transforms for the input images
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Initialize the model
# model = DocumentClassifier()

# # Check if the model file exists
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     model_loaded = True
# else:
#     # If model doesn't exist, we'll use a placeholder approach
#     model_loaded = False
#     print(f"Warning: Model file {model_path} not found. Using fallback approach.")

# @document_classifier_router.post("/classify")
# async def classify_document(file: UploadFile = File(...)):
#     try:
#         # Read and process the uploaded image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')
        
#         if model_loaded:
#             # Use the trained model for classification
#             img_tensor = transform(image).unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 outputs = model(img_tensor)
#                 probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
#                 # Get the predicted class and confidence
#                 predicted_class = torch.argmax(probabilities).item()
#                 confidence = probabilities[predicted_class].item()
                
#                 document_type = DOCUMENT_TYPES.get(predicted_class, "Unknown Document")
#         else:
#             # Fallback approach using image characteristics
#             # This is a simplified version of what we did earlier
#             img_np = np.array(image)
#             height, width = img_np.shape[:2]
#             aspect_ratio = width / height
            
#             # Simple heuristic based on aspect ratio
#             if 1.5 <= aspect_ratio <= 1.6:
#                 document_type = "Aadhaar Card"
#                 confidence = 0.6
#             elif 1.58 <= aspect_ratio <= 1.62:
#                 document_type = "PAN Card"
#                 confidence = 0.6
#             elif aspect_ratio > 1.7:
#                 document_type = "Driving License"
#                 confidence = 0.6
#             else:
#                 document_type = "Identity Document"
#                 confidence = 0.5
        
#         return JSONResponse(content={
#             "document_type": document_type,
#             "confidence": confidence
#         })
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Error in classify_document: {error_details}")
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         )

# @document_classifier_router.post("/ask")
# async def answer_document_question(
#     file: UploadFile = File(...),
#     question: str = Form(...)
# ):
#     try:
#         # First classify the document
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')
        
#         # Reset file position for reuse
#         await file.seek(0)
        
#         # Determine if this is a document identification question
#         question_lower = question.lower()
#         is_identification_question = any(keyword in question_lower for keyword in 
#                                        ["what is this", "what document", "what card", "identify this", "type of"])
        
#         if is_identification_question:
#             # Use our document classifier
#             classification_response = await classify_document(file)
#             classification_data = json.loads(classification_response.body)
            
#             if "error" in classification_data:
#                 raise Exception(classification_data["error"])
            
#             return JSONResponse(content={
#                 "answer": f"This is a {classification_data['document_type']}",
#                 "confidence": classification_data["confidence"]
#             })
#         else:
#             # For other questions about the document, use the ViLT model
#             # Import here to avoid circular imports
#             from api.object_detection import answer_question
#             return await answer_question(file, question)
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Error in answer_document_question: {error_details}")
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         ) 