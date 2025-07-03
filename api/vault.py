from fastapi import APIRouter, HTTPException, Depends, Cookie, Response, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
import time
import json
import os
from datetime import datetime, timedelta
from services.vault import key_vault

router = APIRouter(tags=["vault"])

# Get environment setting
IS_PRODUCTION = os.environ.get('ENVIRONMENT', 'development').lower() == 'production'

class SessionRequest(BaseModel):
    provider: str
    api_key: str
    duration_hours: int = 24
    name: Optional[str] = None
    is_default: Optional[bool] = False

class KeyData(BaseModel):
    id: Optional[str] = None
    name: str
    key: str
    expiration: Optional[str] = None
    isDefault: bool = False
    createdAt: Optional[str] = None

class SaveKeyRequest(BaseModel):
    provider: str
    key_data: KeyData

class DeleteKeyRequest(BaseModel):
    provider: str
    key_id: str

class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key: str  # This will be masked before returning to client
    expiration: Optional[str] = None
    isDefault: bool = False
    createdAt: str

@router.post("/create_session")
async def create_session(request: SessionRequest, response: Response):
    """Create a temporary session for API key usage"""
    try:
        # Create metadata for the session
        metadata = {
            "name": request.name or "Unnamed Key",
            "isDefault": request.is_default,
            "id": str(uuid.uuid4())
        }
        
        # Create a session with the API key
        session_id = key_vault.create_session(
            request.provider, 
            request.api_key, 
            request.duration_hours,
            metadata
        )
        
        # Set a cookie with the session ID (HTTP only for security)
        max_age = request.duration_hours * 3600  # Convert hours to seconds
        response.set_cookie(
            key="api_key_session",
            value=session_id,
            max_age=max_age,
            httponly=True,
            samesite="strict",
            secure=IS_PRODUCTION  # Only require HTTPS in production
        )
        
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/verify_session")
async def verify_session(api_key_session: Optional[str] = Cookie(None)):
    """Verify if the current session is valid"""
    try:
        if not api_key_session:
            return {"valid": False, "message": "No session found"}
            
        is_valid = key_vault.verify_session(api_key_session)
        return {"valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify session: {str(e)}")

@router.post("/save_key")
async def save_key(request: SaveKeyRequest, response: Response, existing_session: str = Cookie(None, alias="api_key_session")):
    """Save an API key as a session, appending to existing keys"""
    try:
        key_data = request.key_data.dict()
        
        # If no ID is provided, generate one
        if not key_data.get("id"):
            key_data["id"] = str(uuid.uuid4())
        
        # If no createdAt is provided, set it to now
        if not key_data.get("createdAt"):
            key_data["createdAt"] = datetime.now().isoformat()
            
        # Calculate expiration in hours from the expiration date, default to 24 hours
        expiration_hours = 24
        if key_data.get("expiration"):
            try:
                expiration_date = datetime.fromisoformat(key_data["expiration"])
                now = datetime.now()
                delta = expiration_date - now
                expiration_hours = max(1, delta.total_seconds() / 3600)  # At least 1 hour
            except Exception:
                # If there's an error parsing the date, use default
                pass
        
        # Create metadata for the session using key data
        metadata = {
            "id": key_data["id"],
            "name": key_data["name"],
            "isDefault": key_data["isDefault"],
            "createdAt": key_data["createdAt"],
            "expiration": key_data.get("expiration"),
            "provider": request.provider  # Ensure provider is included in metadata
        }

        # Check if we have an existing valid session
        if existing_session and existing_session in key_vault.sessions:
            # Get existing session data
            session = key_vault.sessions[existing_session]
            
            # Check if session has expired
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at > datetime.now():
                # Session is still valid, append the new key
                if "keys" not in session:
                    session["keys"] = []
                
                # Check if key with same ID already exists
                existing_key_index = next(
                    (i for i, k in enumerate(session["keys"]) 
                     if k.get("id") == key_data["id"]), None
                )
                
                if existing_key_index is not None:
                    # Update existing key
                    session["keys"][existing_key_index] = {
                        "api_key": key_data["key"],
                        "provider": request.provider,
                        "metadata": metadata,
                        "updated_at": datetime.now().isoformat()
                    }
                else:
                    # Append new key
                    session["keys"].append({
                        "api_key": key_data["key"],
                        "provider": request.provider,
                        "metadata": metadata,
                        "created_at": datetime.now().isoformat()
                    })
                
                # Update session expiration if new expiration is later
                new_expires_at = datetime.now() + timedelta(hours=expiration_hours)
                if new_expires_at > expires_at:
                    session["expires_at"] = new_expires_at.isoformat()
                
                # Save the updated session
                key_vault.save_vault()
                session_id = existing_session
            else:
                # Session expired, create new one
                session_id = key_vault.create_session_with_keys(
                    [{
                        "api_key": key_data["key"],
                        "provider": request.provider,
                        "metadata": metadata,
                        "created_at": datetime.now().isoformat()
                    }],
                    expiration_hours
                )
        else:
            # No existing session, create new one
            session_id = key_vault.create_session_with_keys(
                [{
                    "api_key": key_data["key"],
                    "provider": request.provider,
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat()
                }],
                expiration_hours
            )
        
        # Set a cookie with the session ID (HTTP only for security)
        max_age = int(expiration_hours * 3600)  # Convert hours to seconds
        response.set_cookie(
            key="api_key_session",
            value=session_id,
            max_age=max_age,
            httponly=True,
            samesite="strict",
            secure=IS_PRODUCTION  # Only require HTTPS in production
        )
        
        # For security, mask the key before sending back to client
        response_data = metadata.copy()
        if len(key_data["key"]) > 8:
            response_data["key"] = key_data["key"][:4] + "..." + key_data["key"][-4:]
        else:
            response_data["key"] = "****"
        
        # Add session information
        response_data["session_id"] = session_id
        response_data["session_reused"] = session_id == existing_session
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save key: {str(e)}")

@router.delete("/delete_key")
async def delete_key(request: DeleteKeyRequest):
    """Delete an API key session"""
    try:
        # Get all sessions
        all_sessions = key_vault.get_all_sessions()
        
        # Find the session with the matching provider and key ID
        target_session_id = None
        provider_info = None
        if request.provider in all_sessions:
            for session_id, session_data in all_sessions[request.provider].items():
                if "metadata" in session_data and session_data["metadata"].get("id") == request.key_id:
                    target_session_id = session_id
                    provider_info = request.provider
                    break
        
        if not target_session_id:
            return {"success": False, "message": "Key not found"}
        
        # Delete the session
        success = key_vault.delete_session(target_session_id)
        return {
            "success": success, 
            "session_id": target_session_id,
            "provider": provider_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete key: {str(e)}")

@router.get("/get_keys")
async def get_keys(
    response: Response,
    session_id: str = Cookie(None, alias="api_key_session"),
    provider: str = None
):
    """
    Get all keys for the current session, optionally filtered by provider
    
    :param response: FastAPI response object
    :param session_id: Current session ID from cookie
    :param provider: Optional provider filter
    :return: Dictionary containing keys and metadata
    """
    try:
        # Check if we have an active session
        if not session_id:
            response.status_code = 404
            return {
                "success": False,
                "error": "No active session found",
                "keys": []
            }
        
        # Get session data
        if session_id not in key_vault.sessions:
            response.status_code = 404
            return {
                "success": False,
                "error": "Session not found",
                "keys": []
            }
        
        session = key_vault.sessions[session_id]
        
        # Check session expiration
        try:
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at < datetime.now():
                # Remove expired session
                del key_vault.sessions[session_id]
                key_vault.save_vault()
                response.status_code = 404
                return {
                    "success": False,
                    "error": "Session expired",
                    "keys": []
                }
        except Exception as e:
            response.status_code = 400
            return {
                "success": False,
                "error": f"Invalid session data: {str(e)}",
                "keys": []
            }
        
        # Prepare response data
        result = {
            "success": True,
            "session_id": session_id,
            "expires_at": session["expires_at"],
            "keys": []
        }
        
        # Handle both new format (keys list) and old format (single key)
        if "keys" in session and isinstance(session["keys"], list):
            # New format - multiple keys
            for key_data in session["keys"]:
                key_info = {
                    "provider": key_data.get("provider"),
                    "created_at": key_data.get("created_at"),
                    "updated_at": key_data.get("updated_at"),
                }
                
                # Add metadata if available
                if "metadata" in key_data:
                    metadata = key_data["metadata"]
                    key_info.update({
                        "name": metadata.get("name", "Unnamed Key"),
                        "id": metadata.get("id"),
                        "is_default": metadata.get("isDefault", False),
                        "provider": metadata.get("provider", key_info["provider"]),
                    })
                
                # Check provider filter if specified
                if provider and key_info.get("provider") != provider:
                    continue
                
                # Mask the API key
                api_key = key_data.get("api_key", "")
                if len(api_key) > 8:
                    key_info["masked_key"] = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
                else:
                    key_info["masked_key"] = "****"
                
                # Add expiration information
                key_info["expires_at"] = session["expires_at"]
                
                result["keys"].append(key_info)
                
        elif "api_key" in session:
            # Old format - single key
            key_info = {
                "provider": session.get("provider"),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
            }
            
            # Add metadata if available
            if "metadata" in session:
                metadata = session["metadata"]
                key_info.update({
                    "name": metadata.get("name", "Unnamed Key"),
                    "id": metadata.get("id"),
                    "is_default": metadata.get("isDefault", False),
                    "provider": metadata.get("provider", key_info["provider"]),
                })
            
            # Check provider filter if specified
            session_provider = key_info.get("provider")
            if not provider or session_provider == provider:
                # Mask the API key
                api_key = session["api_key"]
                if len(api_key) > 8:
                    key_info["masked_key"] = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
                else:
                    key_info["masked_key"] = "****"
                
                # Add expiration information
                key_info["expires_at"] = session["expires_at"]
                
                result["keys"].append(key_info)
        
        # Add session metadata if available
        if "metadata" in session:
            result["metadata"] = session["metadata"]
        
        # Add summary information
        result["total_keys"] = len(result["keys"])
        
        # Group keys by provider
        providers = {}
        for key in result["keys"]:
            provider_name = key.get("provider", "unknown")
            if provider_name not in providers:
                providers[provider_name] = 0
            providers[provider_name] += 1
        result["providers"] = providers
        
        return result
        
    except Exception as e:
        response.status_code = 500
        return {
            "success": False,
            "error": f"Failed to get keys: {str(e)}",
            "keys": []
        }

@router.post("/clear_sessions")
async def clear_sessions(response: Response):
    """Clear all session cookies"""
    response.delete_cookie(key="api_key_session")
    
    # Get all providers with active sessions and clear them
    providers = key_vault.clear_all_sessions()
    
    return {
        "success": True,
        "providers": providers
    }

# Export the router with the name that's expected in main.py
vault_router = router 