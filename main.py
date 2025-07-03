from fastapi import FastAPI, Depends, HTTPException, status, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pathlib import Path
from datetime import timedelta

from dotenv import load_dotenv
from services.helper_functions import load_json
from auth.auth import (
    authenticate_user, create_access_token, get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES, User, USERS_DB, create_unlimited_access_token
)

from api.analyze import analyze_logs_router
from api.config import config_router
from api.object_detection import object_detection_router
from api.vault import vault_router
# from api.conference import conference_router
# from api.document_classifier import document_classifier_router

import uvicorn
import os

load_dotenv()

app = FastAPI()

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Only mount static directory if it exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the data directory to serve files
app.mount("/data", StaticFiles(directory="data"), name="data")

# Include the analyze router
app.include_router(
    analyze_logs_router,
    prefix="/api/analyze",
    tags=["analyze"],
    dependencies=[Depends(get_current_user)]
)

# Include the config router
app.include_router(
    config_router,
    prefix="/api/config",
    tags=["config"],
    dependencies=[Depends(get_current_user)]
)

# Include the object detection router
app.include_router(
    object_detection_router,
    prefix="/api",
    tags=["detection"],
    dependencies=[Depends(get_current_user)]
)

# Include the vault router
app.include_router(
    vault_router,
    prefix="/api/vault",
    tags=["vault"],
    dependencies=[Depends(get_current_user)]
)


# from demo.routes.wrapper import demo_router  # ✅ Import this

# app.include_router(
#     demo_router,
#     prefix="/demo",
#     tags=["demo"],
#     dependencies=[Depends(get_current_user)]
# )

# Include the conference router
# app.include_router(conference_router, prefix="/api/conference", tags=["conference"])

# Include the document classifier router
# app.include_router(document_classifier_router, prefix="/api/documents", tags=["documents"])

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(USERS_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/unlimited-token/{stub}")
async def create_unlimited_token(stub: str):
    """
    Create an unlimited access token (no expiration) with a stub identifier.
    
    Args:
        stub: A unique identifier for the token
    
    Returns:
        dict: Contains the unlimited access token and token type
    """
    access_token = create_unlimited_access_token(
        data={"sub": f"internal_{stub}"}
    )
    return {"access_token": access_token, "token_type": "bearer", "stub": stub}

# Root route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        # Try to get the Authorization header
        # auth_header = request.headers.get('Authorization')
        # if not auth_header or not auth_header.startswith('Bearer '):
            # return RedirectResponse(url='/login', status_code=302)
        
        # If we have a token, try to validate it
        # token = auth_header.split(' ')[1]
        # user = await get_current_user(token)
        
        config = load_json("json/config.json")
        task_types = [task_type for task_type in config.keys() if task_type != 'models']
        models = set()
        
        # Iterate through task types and extract models
        for task_type in task_types:
            for task in config[task_type]:
                if isinstance(task, dict) and "model" in task:
                    models.add(task["model"])
        
        # Add models from the models configuration if it exists
        if "models" in config and isinstance(config["models"], dict):
            models_dict = {
                model_id: f"{model_id} ({info['provider']}) "
                for model_id, info in config["models"].items()
            }
        
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "task_types": task_types, "models": models_dict}
        )
    except HTTPException:
        return RedirectResponse(url='/login', status_code=302)

@app.get("/kyc", response_class=HTMLResponse)
async def kyc_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "kyc.html",
        {"request": request}
    )

@app.get("/kyc/admin", response_class=HTMLResponse)
async def kyc_admin_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "kyc_admin.html",
        {"request": request}
    )

@app.get("/kyc/{session_id}", response_class=HTMLResponse)
async def kyc_session_page(request: Request, session_id: str, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "kyc.html",
        {"request": request, "session_id": session_id}
    )

# @app.get("/conference", response_class=HTMLResponse)
# async def conference_page(request: Request):
#     return templates.TemplateResponse(
#         "conference.html",
#         {"request": request}
#     )


@app.get("/key-management", response_class=HTMLResponse)
async def key_management_page(request: Request):
    return templates.TemplateResponse(
        "key_management.html",
        {"request": request}
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict to specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)