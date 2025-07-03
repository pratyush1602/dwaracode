import json
import os
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import base64
import logging
import uuid

logger = logging.getLogger(__name__)

class KeyVault:
    def __init__(self, vault_file="vault/sessions.vault"):
        self.vault_file = vault_file
        self.vault_dir = os.path.dirname(vault_file)
        self.sessions = {}
        self._ensure_vault_exists()
        self._load_encryption_key()
        self.load_vault()

    def _ensure_vault_exists(self):
        """Ensure vault directory and files exist"""
        if not os.path.exists(self.vault_dir):
            os.makedirs(self.vault_dir)
        
        # Create encryption key file if it doesn't exist
        key_file = os.path.join(self.vault_dir, ".key")
        if not os.path.exists(key_file):
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)

    def _load_encryption_key(self):
        """Load or create encryption key"""
        key_file = os.path.join(self.vault_dir, ".key")
        with open(key_file, "rb") as f:
            key = f.read()
        self.cipher = Fernet(key)

    def load_vault(self):
        """Load encrypted vault contents (only sessions, not permanent keys)"""
        try:
            if os.path.exists(self.vault_file):
                with open(self.vault_file, "rb") as f:
                    encrypted_data = f.read()
                    if encrypted_data:
                        decrypted_data = self.cipher.decrypt(encrypted_data)
                        vault_data = json.loads(decrypted_data)
                        self.sessions = vault_data.get("sessions", {})
                        # Clean expired sessions
                        self._clean_expired_sessions()
        except Exception as e:
            logger.error(f"Error loading vault: {str(e)}")
            self.sessions = {}

    def save_vault(self):
        """Save encrypted vault contents (only sessions)"""
        try:
            vault_data = {
                "sessions": self.sessions
            }
            encrypted_data = self.cipher.encrypt(json.dumps(vault_data).encode())
            with open(self.vault_file, "wb") as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Error saving vault: {str(e)}")
            raise

    def create_session_with_keys(self, keys: list, expiration_hours: float = 24) -> str:
        """
        Create a new session with multiple keys
        
        :param keys: List of dictionaries containing key data
                    Each dict should have: api_key, provider, metadata, created_at
        :param expiration_hours: Number of hours until expiration
        :return: Session ID
        """
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=expiration_hours)
        
        # Create the session with the keys list
        self.sessions[session_id] = {
            "keys": keys.copy(),  # Create a copy of the keys list
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        # If there's only one key, also store its metadata at the session level for backwards compatibility
        if len(keys) == 1:
            first_key = keys[0]
            self.sessions[session_id].update({
                "api_key": first_key["api_key"],
                "provider": first_key["provider"],
                "metadata": first_key["metadata"]
            })
        
        # Save changes to vault file
        self.save_vault()
        
        return session_id

    def create_session(self, provider: str, api_key: str, expiration_hours: float = 24, metadata: dict = None) -> str:
        """
        Create a new session (wrapper for backwards compatibility)
        
        :param provider: The provider name
        :param api_key: The API key
        :param expiration_hours: Number of hours until expiration
        :param metadata: Optional metadata dictionary
        :return: Session ID
        """
        # Create a key entry in the new format
        key_entry = {
            "api_key": api_key,
            "provider": provider,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        # Create session using the new method
        return self.create_session_with_keys([key_entry], expiration_hours)

    def get_session_key(self, session_id: str, provider: str = None) -> tuple:
        """
        Get API key for a session, optionally filtering by provider
        
        :param session_id: The session ID to retrieve
        :param provider: Optional provider name to ensure key matches
        :return: (api_key, provider, metadata) tuple
        """
        session = self.sessions[session_id]
        for key in session["keys"]:
            if provider and key["provider"] != provider:
                continue
            return key["api_key"], key["provider"], key.get("metadata", {})
        return None, None, "Session not found"
        
        session = self.sessions[session_id]
        expires_at = datetime.fromisoformat(session["expires_at"])
        
        # Check if session has expired
        if expires_at < datetime.now():
            del self.sessions[session_id]
            self.save_vault()
            return None, None, "Session expired"
        
        # Get provider from session directly or from metadata
        session_provider = session.get("provider")
        if not session_provider and "metadata" in session and "provider" in session["metadata"]:
            session_provider = session["metadata"]["provider"]
        
        # If provider is specified, ensure it matches
        if provider and session_provider and provider != session_provider:
            return None, None, f"Session key is for provider {session_provider}, not {provider}"
        
        return session["api_key"], session_provider, session.get("metadata", {})

    def verify_session(self, session_id: str) -> bool:
        """Verify if a session is valid"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        expires_at = datetime.fromisoformat(session["expires_at"])
        
        if expires_at < datetime.now():
            del self.sessions[session_id]
            self.save_vault()
            return False
        
        return True

    def get_all_sessions(self, include_keys=False):
        """Get all active sessions"""
        result = {}
        now = datetime.now()
        
        for session_id, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at >= now:
                session_info = {
                    "provider": session["provider"],
                    "expires_at": session["expires_at"],
                    "created_at": session.get("created_at", "Unknown"),
                }
                
                # Include metadata if present
                if "metadata" in session:
                    session_info["metadata"] = session["metadata"]
                
                # Only include the actual key if requested (and mask it)
                if include_keys and "api_key" in session:
                    key = session["api_key"]
                    if len(key) > 8:
                        session_info["key"] = key[:4] + "..." + key[-4:]
                    else:
                        session_info["key"] = "****"
                
                # Group by provider
                provider = session["provider"]
                if provider not in result:
                    result[provider] = {}
                
                result[provider][session_id] = session_info
        
        return result

    def _clean_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at < now:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        if expired:
            self.save_vault()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_vault()
            return True
        return False

    def clear_all_sessions(self):
        """Remove all active sessions"""
        # Get list of providers before clearing
        providers = list(self.get_all_sessions().keys())
        
        # Clear all sessions
        self.sessions = {}
        self.save_vault()
        
        return providers

    def update_session(self, session_id: str, provider: str, api_key: str, 
                      expiration_hours: float = 24, metadata: dict = None) -> str:
        """
        Update an existing session with new data
        
        :param session_id: The session ID to update
        :param provider: The provider name
        :param api_key: The API key
        :param expiration_hours: Number of hours until expiration
        :param metadata: Optional metadata dictionary
        :return: The session ID
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Calculate expiration time
        expires_at = datetime.now() + timedelta(hours=expiration_hours)
        
        # Update the session data
        self.sessions[session_id] = {
            "provider": provider,
            "api_key": api_key,
            "expires_at": expires_at.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add metadata if provided
        if metadata:
            self.sessions[session_id]["metadata"] = metadata
        
        # Preserve original creation time if it exists
        if "created_at" in self.sessions[session_id]:
            self.sessions[session_id]["created_at"] = self.sessions[session_id]["created_at"]
        else:
            self.sessions[session_id]["created_at"] = datetime.now().isoformat()
        
        # Save changes to vault file
        self.save_vault()
        
        return session_id

    def is_valid_session(self, session_id: str) -> bool:
        """
        Check if a session is valid and not expired
        
        :param session_id: The session ID to check
        :return: True if session is valid, False otherwise
        """
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if "expires_at" not in session:
            return False
        
        try:
            expires_at = datetime.fromisoformat(session["expires_at"])
            return expires_at > datetime.now()
        except Exception:
            return False

    def get_all_session_keys(self, session_id: str) -> dict:
        """
        Get all keys for a session with their metadata
        
        :param session_id: The session ID to retrieve
        :return: Dictionary containing all session keys and metadata
        """
        if not session_id or session_id not in self.sessions:
            return {
                "success": False,
                "error": "Session not found",
                "keys": []
            }
        
        session = self.sessions[session_id]
        
        # Check expiration
        try:
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at < datetime.now():
                del self.sessions[session_id]
                self.save_vault()
                return {
                    "success": False,
                    "error": "Session expired",
                    "keys": []
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid session data: {str(e)}",
                "keys": []
            }
        
        # Initialize response
        result = {
            "success": True,
            "session_id": session_id,
            "expires_at": session["expires_at"],
            "created_at": session.get("created_at", "Unknown"),
            "keys": []
        }
        
        # Handle both new and old format sessions
        if "keys" in session and isinstance(session["keys"], list):
            # New format - multiple keys
            for key in session["keys"]:
                key_info = self._format_key_info(key)
                if key_info:
                    result["keys"].append(key_info)
        else:
            # Old format - single key
            key_info = self._format_key_info(session)
            if key_info:
                result["keys"].append(key_info)
        
        # Group keys by provider
        keys_by_provider = {}
        for key in result["keys"]:
            provider = key.get("provider", "unknown")
            if provider not in keys_by_provider:
                keys_by_provider[provider] = []
            keys_by_provider[provider].append(key)
        
        # Sort keys within each provider group
        for provider in keys_by_provider:
            keys_by_provider[provider].sort(
                key=lambda k: (
                    not k.get("isDefault", False),  # Default keys first
                    k.get("created_at", "")  # Then by creation date
                )
            )
        
        result["keys_by_provider"] = keys_by_provider
        result["total_keys"] = len(result["keys"])
        result["provider_count"] = len(keys_by_provider)
        
        return result

    def _format_key_info(self, key_data: dict) -> dict:
        """
        Format key information consistently
        
        :param key_data: Raw key data from session
        :return: Formatted key information
        """
        try:
            # Get metadata
            metadata = key_data.get("metadata", {})
            if not metadata and "api_key" not in key_data:
                metadata = key_data  # Old format might have metadata at root
            
            # Build key info
            key_info = {
                "id": metadata.get("id") or str(uuid.uuid4()),
                "name": metadata.get("name", "Unnamed Key"),
                "provider": (
                    key_data.get("provider") or 
                    metadata.get("provider") or 
                    "unknown"
                ),
                "isDefault": metadata.get("isDefault", False),
                "created_at": (
                    key_data.get("created_at") or 
                    metadata.get("createdAt") or 
                    "Unknown"
                ),
                "updated_at": key_data.get("updated_at"),
            }
            
            # Add expiration if available
            if "expiration" in metadata:
                key_info["expiration"] = metadata["expiration"]
                
                # Add expiration status
                try:
                    exp_date = datetime.fromisoformat(metadata["expiration"])
                    key_info["expired"] = exp_date < datetime.now()
                    key_info["expires_in"] = (exp_date - datetime.now()).total_seconds()
                except:
                    key_info["expired"] = False
            
            # Mask the API key
            api_key = key_data.get("api_key", "")
            if api_key:
                if len(api_key) > 8:
                    key_info["masked_key"] = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
                else:
                    key_info["masked_key"] = "****"
            
            return key_info
        except Exception as e:
            logger.error(f"Error formatting key info: {e}")
            return None

# Global vault instance
key_vault = KeyVault()

# For backward compatibility with code that imports 'vault'
vault = key_vault 