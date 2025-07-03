import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_json(path: str):
    file_path = Path(path)
    with open(file_path) as f:
        return json.load(f)
    
def create_system_prompt(system_prompt_json: dict) -> str:
    """
    Creates a system prompt from the provided JSON configuration.
    Uses default values if keys are missing.
    
    Args:
        system_prompt_json (dict): Dictionary containing prompt configuration
        
    Returns:
        str: Formatted system prompt
    """
    try:
        # Default configuration
        default_config = {
            "description": "You are a versatile AI assistant designed to help with various tasks.",
            "core_capabilities": {
                "1": "Multi-domain Expertise: Handle various types of tasks",
                "2": "Pattern Recognition: Identify patterns and key information",
                "3": "Problem Analysis: Break down complex problems"
            },
            "analysis_approach": {
                "1": "Be Precise: Base analysis on provided information",
                "2": "Be Structured: Organize responses clearly",
                "3": "Be Transparent: Indicate limitations when needed"
            },
            "limitations": {
                "1": "Cannot access external systems",
                "2": "Cannot execute code",
                "3": "Cannot make guarantees without context"
            }
        }

        # Get prompt configuration with fallbacks
        prompt_config = system_prompt_json.get('1', {})
        
        # Use get() with default values from default_config
        description = prompt_config.get('description', default_config['description'])
        core_capabilities = prompt_config.get('core_capabilities', default_config['core_capabilities'])
        analysis_approach = prompt_config.get('analysis_approach', default_config['analysis_approach'])
        limitations = prompt_config.get('limitations', default_config['limitations'])

        # Create the system prompt using available values
        system_prompt = f"""
{description}

## Core Capabilities:
{chr(10).join([f"{k}. {v}" for k, v in core_capabilities.items()])}

## Analysis Approach:
{chr(10).join([f"{k}. {v}" for k, v in analysis_approach.items()])}

## Limitations:
{chr(10).join([f"{k}. {v}" for k, v in limitations.items()])}
"""
        logger.info("System prompt created successfully")
        return system_prompt.strip()

    except Exception as e:
        logger.warning(f"Error creating system prompt: {str(e)}. Using default configuration.")
        # If anything goes wrong, return a basic default prompt
        return """
You are a versatile AI assistant designed to help with various tasks.

## Core Capabilities:
1. Multi-domain Expertise: Handle various types of tasks
2. Pattern Recognition: Identify patterns and key information
3. Problem Analysis: Break down complex problems

## Analysis Approach:
1. Be Precise: Base analysis on provided information
2. Be Structured: Organize responses clearly
3. Be Transparent: Indicate limitations when needed

## Limitations:
1. Cannot access external systems
2. Cannot execute code
3. Cannot make guarantees without context
""".strip()
