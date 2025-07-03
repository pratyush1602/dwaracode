from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from datetime import datetime
import glob
import os

config_router = APIRouter()

@config_router.get("/")
async def get_config():
    """Get the current configuration"""
    try:
        with open("json/config.json", "r") as f:
            config = json.load(f)
        return {"status": "success", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@config_router.get("/history")
async def get_config_history():
    """Get configuration history"""
    try:
        # Get all backup files sorted by timestamp (newest first)
        backup_files = glob.glob("json/config_backup_*.json")
        backup_files.sort(reverse=True)
        
        history = []
        for file in backup_files:
            try:
                with open(file, "r") as f:
                    config = json.load(f)
                    # Extract timestamp from filename
                    timestamp = os.path.basename(file).replace("config_backup_", "").replace(".json", "")
                    formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat()
                    
                    history.append({
                        "timestamp": formatted_timestamp,
                        "config": config,
                        "filename": os.path.basename(file)
                    })
            except Exception as e:
                print(f"Error reading backup file {file}: {str(e)}")
                continue
                
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/update")
async def update_config(config: dict):
    """Update the configuration"""
    try:
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"json/config_backup_{timestamp}.json")
        
        # Read and backup existing config
        config_path = Path("json/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                existing_config = json.load(f)
                # Save backup with timestamp
                with open(backup_path, "w") as bf:
                    json.dump(existing_config, bf, indent=2)
        
        # Save new config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        return {
            "status": "success", 
            "message": f"Configuration updated successfully. Backup created at {backup_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/models/update")
async def update_models(model_data: dict):
    """Update the models section of the configuration"""
    try:
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"json/config_backup_{timestamp}.json")
        
        # Read existing config
        config_path = Path("json/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                # Save backup
                with open(backup_path, "w") as bf:
                    json.dump(config, bf, indent=2)
        else:
            config = {}
        
        # Update only the models section
        config["models"] = model_data
        
        # Save updated config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        return {
            "status": "success", 
            "message": f"Models configuration updated successfully. Backup created at {backup_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 