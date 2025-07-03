from dotenv import load_dotenv
import os
import json
import asyncio
import base64
import io
import logging
import importlib
from PIL import Image
from services.helper_functions import load_json, create_system_prompt
from services.vault import vault

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load system prompts
system_prompt_json = load_json("json/system_prompts.json")

# Provider registry - will be populated dynamically
PROVIDER_REGISTRY = {}

# Model registry - maps models to their respective providers and implementation details
# MODEL_REGISTRY = {}  # Initialize as empty dict
# try:
#     MODEL_REGISTRY = build_model_registry_from_config()
# except Exception as e:
#     logger.error(f"Initial model registry build failed: {str(e)}")

# Model registry - maps models to their respective providers and implementation details
MODEL_REGISTRY = {
    # OpenAI Models
    "gpt-4o-mini": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 8192
    },
    "gpt-4o": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 128000
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 128000
    },
    "gpt-4": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 8192
    },
    "gpt-4-vision-preview": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 128000
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "vision_capable": True,
        "max_tokens": 16385
    },
    
    # Anthropic Models
    "claude-3-opus": {
        "provider": "anthropic",
        "vision_capable": True,
        "max_tokens": 200000
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "vision_capable": True,
        "max_tokens": 200000
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "vision_capable": True,
        "max_tokens": 200000
    },
    "claude-2.1": {
        "provider": "anthropic",
        "vision_capable": True,
        "max_tokens": 100000
    },
    "claude-2.0": {
        "provider": "anthropic",
        "vision_capable": True,
        "max_tokens": 100000
    },
    
    # Google Models
    "gemini-1.5-pro": {
        "provider": "google",
        "vision_capable": True,
        "max_tokens": 1000000
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "vision_capable": True,
        "max_tokens": 1000000
    },
    "gemini-2.0-flash-exp": {
        "provider": "google",
        "vision_capable": True,
        "max_tokens": 2048
    },
    "gemini-pro": {
        "provider": "google",
        "vision_capable": True,
        "max_tokens": 32768
    },
    "gemini-pro-vision": {
        "provider": "google",
        "vision_capable": True,
        "max_tokens": 16384
    },
    
    # DeepInfra Models
    "deepinfra/meta-llama/Llama-2-70b-chat-hf": {
        "provider": "deepinfra",
        "vision_capable": False,
        "max_tokens": 4096
    },
    "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "provider": "deepinfra",
        "vision_capable": False,
        "max_tokens": 4096
    },
    "deepinfra/meta-llama/Llama-2-13b-chat-hf": {
        "provider": "deepinfra",
        "vision_capable": False,
        "max_tokens": 4096
    }
}

class ProviderInterface:
    """Base interface that all providers must implement"""
    
    @classmethod
    async def initialize(cls, api_key):
        """Initialize the provider with the given API key"""
        raise NotImplementedError("Provider must implement initialize method")
    
    @classmethod
    async def generate_text(cls, system_prompt, user_prompt, model, **kwargs):
        """Generate text response"""
        raise NotImplementedError("Provider must implement generate_text method")
    
    @classmethod
    async def analyze_image(cls, image_path, prompt, model, **kwargs):
        """Analyze image"""
        raise NotImplementedError("Provider must implement analyze_image method")

class OpenAIProvider(ProviderInterface):
    """OpenAI provider implementation"""
    client = None
    
    @classmethod
    async def initialize(cls, api_key):
        try:
            from openai import AsyncOpenAI
            cls.client = AsyncOpenAI(api_key=api_key)
            return True
        except ImportError:
            logger.warning("OpenAI package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {str(e)}")
            return False
    
    @classmethod
    async def generate_text(cls, system_prompt, user_prompt, model, **kwargs):
        try:
            response = await cls.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=kwargs.get('temperature', 0)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    @classmethod
    async def analyze_image(cls, image_path, prompt, model, **kwargs):
        try:
            # Check if we're dealing with a PDF (either by path or by passed data)
            is_pdf = image_path.lower().endswith('.pdf') or 'pdf_data' in kwargs
            
            if is_pdf:
                # Handle PDF file
                if 'pdf_base64' in kwargs:
                    # Use provided base64 data
                    pdf_base64 = kwargs['pdf_base64']
                else:
                    # Read and encode the PDF
                    with open(image_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
                
                response = await cls.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "file_attachment",
                                    "file_attachment": {
                                        "type": "application/pdf",
                                        "data": pdf_base64
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=kwargs.get('max_tokens', 2048)
                )
            else:
                # Handle image file (existing code)
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = await cls.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=kwargs.get('max_tokens', 2048)
                )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI Vision/PDF API error: {str(e)}")
            raise

class AnthropicProvider(ProviderInterface):
    """Anthropic provider implementation"""
    client = None
    
    @classmethod
    async def initialize(cls, api_key):
        try:
            import anthropic
            cls.client = anthropic.Anthropic(api_key=api_key)
            return True
        except ImportError:
            logger.warning("Anthropic package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {str(e)}")
            return False
    
    @classmethod
    async def generate_text(cls, system_prompt, user_prompt, model, **kwargs):
        try:
            response = await cls.client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    @classmethod
    async def analyze_image(cls, image_path, prompt, model, **kwargs):
        try:
            # Check if we're dealing with a PDF
            is_pdf = image_path.lower().endswith('.pdf') or 'pdf_data' in kwargs
            
            if is_pdf:
                # Handle PDF file
                if 'pdf_data' in kwargs:
                    # Use provided PDF data
                    pdf_data = kwargs['pdf_data']
                else:
                    # Read the PDF
                    with open(image_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                
                # Encode the PDF
                pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
                
                content = [
                    {
                        "type": "file",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            else:
                # Handle image file (existing code)
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            
            response = await cls.client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 2048),
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic Vision/PDF API error: {str(e)}")
            raise

class GoogleProvider(ProviderInterface):
    """Google provider implementation"""
    client = None
    
    @classmethod
    async def initialize(cls, api_key):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            cls.client = genai
            return True
        except ImportError:
            logger.warning("Google Generative AI package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Google: {str(e)}")
            return False
    
    @classmethod
    async def generate_text(cls, system_prompt, user_prompt, model, **kwargs):
        try:
            gemini_model = cls.client.GenerativeModel(model)
            
            combined_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}"
            
            response = gemini_model.generate_content(
                combined_prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.2),
                    'top_p': kwargs.get('top_p', 0.8),
                    'top_k': kwargs.get('top_k', 40),
                    'max_output_tokens': kwargs.get('max_tokens', 2048),
                }
            )
            
            if not response.text:
                return "No response generated"
                
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            raise
    
    @classmethod
    async def analyze_image(cls, image_path, prompt, model, **kwargs):
        try:
            # Check if we're dealing with a PDF
            is_pdf = image_path.lower().endswith('.pdf')
            
            if is_pdf:
                # For PDFs, we need PyMuPDF to convert to images
                try:
                    import fitz  # PyMuPDF
                    
                    # Open the PDF
                    pdf_document = fitz.open(image_path)
                    
                    # For simplicity, we'll analyze the first page
                    page = pdf_document[0]
                    
                    # Render page to an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Add page info to prompt
                    enhanced_prompt = f"{prompt} (Page 1 of {len(pdf_document)})"
                    
                    # Use the image for analysis
                    vision_model = cls.client.GenerativeModel(model)
                    response = vision_model.generate_content(
                        [enhanced_prompt, image],
                        generation_config={
                            'temperature': kwargs.get('temperature', 0.7),
                            'top_p': kwargs.get('top_p', 0.8),
                            'top_k': kwargs.get('top_k', 40),
                            'max_output_tokens': kwargs.get('max_tokens', 2048),
                        }
                    )
                    
                    # Add note about limited analysis
                    if len(pdf_document) > 1:
                        return f"{response.text}\n\nNote: Only analyzed the first page of {len(pdf_document)} total pages."
                    else:
                        return response.text
                        
                except ImportError:
                    return "PDF analysis requires PyMuPDF (fitz) library. Please install it with: pip install pymupdf"
            else:
                # Handle regular image (existing code)
                image = Image.open(image_path)
                
                vision_model = cls.client.GenerativeModel(model)
                
                response = vision_model.generate_content(
                    [prompt, image],
                    generation_config={
                        'temperature': kwargs.get('temperature', 0.7),
                        'top_p': kwargs.get('top_p', 0.8),
                        'top_k': kwargs.get('top_k', 40),
                        'max_output_tokens': kwargs.get('max_tokens', 2048),
                    }
                )
                
                return response.text
        except Exception as e:
            logger.error(f"Google Vision/PDF API error: {str(e)}")
            raise

class DeepInfraProvider(ProviderInterface):
    """DeepInfra provider implementation"""
    client = None
    current_model = None
    api_key = None
    
    @classmethod
    async def initialize(cls, api_key):
        try:
            # Try different import paths
            try:
                from langchain_community.chat_models import ChatDeepInfra
            except ImportError:
                # Fallback to older versions
                from langchain.chat_models import ChatDeepInfra
            
            import os
            
            # Store the API key
            cls.api_key = api_key
            os.environ["DEEPINFRA_API_TOKEN"] = api_key
            
            cls.current_model = "google/gemma-2-9b-it"
            cls.client = ChatDeepInfra(
                api_key=api_key,
                model_name=cls.current_model,
                deepinfra_api_token=api_key
            )
            return True
        except ImportError as e:
            logger.warning(f"Required packages not installed: {str(e)}. Please install with: pip install langchain-community")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize DeepInfra: {str(e)}")
            return False
    
    @classmethod
    async def generate_text(cls, system_prompt, user_prompt, model, **kwargs):
        try:
            # Import here as well to ensure availability
            try:
                from langchain_community.chat_models import ChatDeepInfra
            except ImportError:
                from langchain.chat_models import ChatDeepInfra
                
            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # Update the model if different from current
            if cls.current_model != model:
                logger.info(f"Switching DeepInfra model from {cls.current_model} to {model}")
                cls.current_model = model
                cls.client = ChatDeepInfra(
                    api_key=cls.api_key,
                    model_name=model,
                    deepinfra_api_token=cls.api_key
                )
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Generate response
            try:
                response = await cls.client.ainvoke(messages)
            except AttributeError:
                # Fallback to synchronous invoke if ainvoke is not available
                response = await asyncio.to_thread(cls.client.invoke, messages)
            
            parser = StrOutputParser()
            return parser.invoke(response.content)
            
        except ImportError as e:
            error_msg = f"Required packages not installed: {str(e)}. Please install with: pip install langchain-community"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            logger.error(f"DeepInfra API error: {str(e)}")
            if "Not authenticated" in str(e):
                return "Authentication failed. Please check your DeepInfra API key."
            raise
    
    @classmethod
    async def analyze_image(cls, image_path, prompt, model, **kwargs):
        """DeepInfra does not currently support image analysis"""
        return "Image analysis is not supported by DeepInfra models."

# Register built-in providers
PROVIDER_REGISTRY["openai"] = OpenAIProvider
PROVIDER_REGISTRY["anthropic"] = AnthropicProvider
PROVIDER_REGISTRY["google"] = GoogleProvider
PROVIDER_REGISTRY["deepinfra"] = DeepInfraProvider

# Function to dynamically load additional providers from a directory
def load_providers_from_directory(directory_path="services/providers"):
    """Load provider modules from a directory"""
    if not os.path.exists(directory_path):
        logger.info(f"Provider directory {directory_path} does not exist. Skipping.")
        return
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove .py extension
            try:
                module = importlib.import_module(f"services.providers.{module_name}")
                
                # Look for a provider class that inherits from ProviderInterface
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, ProviderInterface) and 
                        attr is not ProviderInterface):
                        
                        # Get provider name from the class or module
                        provider_name = getattr(attr, "PROVIDER_NAME", module_name.lower())
                        
                        # Register the provider
                        PROVIDER_REGISTRY[provider_name] = attr
                        logger.info(f"Registered provider: {provider_name}")
            except Exception as e:
                logger.error(f"Error loading provider module {module_name}: {str(e)}")

# Initialize providers
async def initialize_provider(provider_name, session_id=None):
    """Initialize a specific provider with either env var or vault session"""
    if provider_name not in PROVIDER_REGISTRY:
        logger.warning(f"Provider '{provider_name}' not found in registry")
        return False
    
    # First try session_id if provided - now passing provider_name to filter
    if session_id:
        api_key, provider, error_message = vault.get_session_key(session_id, provider_name)
        
        if api_key:
            logger.info(f"Initializing {provider_name} using session key")
            return await PROVIDER_REGISTRY[provider_name].initialize(api_key)
        else:
            logger.warning(f"No valid session key found: {error_message}")
    
    # Fallback to environment variable
    api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
    print(f"api_key: {api_key}")
    if not api_key:
        logger.warning(f"API key for provider '{provider_name}' is missing")
        return False
    
    return await PROVIDER_REGISTRY[provider_name].initialize(api_key)

# Get available models
def get_available_models():
    """Get a list of all available models based on provider availability"""
    available_models = []
    initialized_providers = set()
    
    for model_name, model_info in MODEL_REGISTRY.items():
        provider = model_info["provider"]
        
        # Skip if we already checked this provider and it's not available
        if provider in initialized_providers:
            continue
        
        # Try to initialize the provider
        if provider in PROVIDER_REGISTRY:
            asyncio.run(initialize_provider(provider))
            initialized_providers.add(provider)
            
            # Add all models for this provider
            for m_name, m_info in MODEL_REGISTRY.items():
                if m_info["provider"] == provider:
                    available_models.append(m_name)
    
    return available_models

# Get fallback model
def get_fallback_model():
    """Get a fallback model based on available providers"""
    for provider in ["openai", "anthropic", "google"]:
        if provider in PROVIDER_REGISTRY:
            if asyncio.run(initialize_provider(provider)):
                # Find the first model for this provider
                for model_name, model_info in MODEL_REGISTRY.items():
                    if model_info["provider"] == provider:
                        return model_name
    
    raise RuntimeError("No LLM providers are available. Please install at least one provider package.")

# Generate response
async def generate_response(prompt: str, model: str = "gpt-4o-mini", session_id=None, **kwargs):
    """Generate a response using the specified model"""
    global MODEL_REGISTRY  # Add this line to declare MODEL_REGISTRY as global
    
    system_prompt = create_system_prompt(system_prompt_json)
    
    # Check if model exists in registry
    if model not in MODEL_REGISTRY:
        logger.warning(f"Model '{model}' not found in registry. Attempting to rebuild registry.")
        try:
            MODEL_REGISTRY = build_model_registry_from_config()
            if model not in MODEL_REGISTRY:
                logger.error(f"Model '{model}' still not found after rebuilding registry.")
                raise ValueError(f"Model '{model}' not found in registry. Please check your configuration.")
        except Exception as e:
            logger.error(f"Error rebuilding model registry: {str(e)}")
            raise ValueError(f"Failed to rebuild model registry: {str(e)}")
    
    # Get provider for this model
    model_info = MODEL_REGISTRY[model]
    provider_name = model_info["provider"]
    
    # Check if provider exists
    if provider_name not in PROVIDER_REGISTRY:
        logger.warning(f"Provider '{provider_name}' not found in registry. Using fallback model.")
        # model = get_fallback_model()
        model_info = MODEL_REGISTRY[model]
        provider_name = model_info["provider"]
    
    # Initialize provider if needed
    if not await initialize_provider(provider_name, session_id):
        logger.warning(f"Provider '{provider_name}' initialization failed. Using fallback model.")
        # model = get_fallback_model()
        model_info = MODEL_REGISTRY[model]
        provider_name = model_info["provider"]
        
        # If we still can't initialize, give up
        if not await initialize_provider(provider_name):
            return "Failed to initialize any provider. Please check your API keys."
    
    logger.info(f"Generating response using model: {model} (provider: {provider_name})")
    
    # Call the provider's generate_text method
    try:
        provider = PROVIDER_REGISTRY[provider_name]
        return await provider.generate_text(system_prompt, prompt, model, **kwargs)
    except Exception as e:
        logger.error(f"Error generating response with {model}: {str(e)}")
        return f"Error generating response: {str(e)}"

# Analyze image
async def analyze_image_with_vision_model(image_path: str, prompt: str, model: str = None, session_id=None, **kwargs):
    """Analyze an image using a vision-capable model"""
    # If no model specified, find the first available vision model
    global MODEL_REGISTRY
    if model is None:
        for model_name, model_info in MODEL_REGISTRY.items():
            if model_info["vision_capable"]:
                provider_name = model_info["provider"]
                if provider_name in PROVIDER_REGISTRY and await initialize_provider(provider_name, session_id):
                    model = model_name
                    break
        
        if model is None:
            return "No vision-capable models are available."
    
    # Check if model exists and is vision-capable
    if model not in MODEL_REGISTRY:
        logger.warning(f"Model '{model}' not found in registry. Using fallback model.")
        # model = get_fallback_model()
        MODEL_REGISTRY=build_model_registry_from_config()
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model}' not found in registry. Please check your configuration.")
    
    model_info = MODEL_REGISTRY[model]
    if not model_info["vision_capable"]:
        return f"Model '{model}' is not vision-capable."
    
    # Get provider for this model
    provider_name = model_info["provider"]
    
    # Check if provider exists
    if provider_name not in PROVIDER_REGISTRY:
        return f"Provider '{provider_name}' not found in registry."
    
    # Initialize provider if needed - pass the session_id
    if not await initialize_provider(provider_name, session_id):
        # Fallback to environment variable failed
        return f"Provider '{provider_name}' initialization failed. Unable to use API key from session or environment."
    
    logger.info(f"Analyzing image using model: {model} (provider: {provider_name})")
    
    # Call the provider's analyze_image method
    try:
        provider = PROVIDER_REGISTRY[provider_name]
        return await provider.analyze_image(image_path, prompt, model, **kwargs)
    except Exception as e:
        logger.error(f"Error analyzing image with {model}: {str(e)}")
        return f"Error analyzing image: {str(e)}"

# Add this new function for PDF analysis
async def analyze_pdf_with_vision_model(pdf_path: str, prompt: str, model: str = None, session_id=None, **kwargs):
    """
    Analyze a PDF document using a vision-capable model.
    
    :param pdf_path: Path to the PDF file
    :param prompt: The prompt to use for analysis
    :param model: The model to use for analysis (must be vision-capable)
    :param session_id: The session ID to use for authentication
    :return: The analysis result
    """
    # If no model specified, find the first available vision model
    global MODEL_REGISTRY
    if model is None:
        for model_name, model_info in MODEL_REGISTRY.items():
            if model_info["vision_capable"]:
                provider_name = model_info["provider"]
                if provider_name in PROVIDER_REGISTRY and await initialize_provider(provider_name, session_id):
                    model = model_name
                    break
        
        if model is None:
            return "No vision-capable models are available for PDF analysis."
    
    # Check if model exists and is vision-capable
    if model not in MODEL_REGISTRY:
        logger.warning(f"Model '{model}' not found in registry. Using fallback model.")
        # model = get_fallback_model()
        MODEL_REGISTRY=build_model_registry_from_config()
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model}' not found in registry. Please check your configuration.")
    
    model_info = MODEL_REGISTRY[model]
    if not model_info["vision_capable"]:
        return f"Model '{model}' is not vision-capable and cannot analyze PDFs visually."
    
    # Get provider for this model
    provider_name = model_info["provider"]
    
    # Check if provider exists
    if provider_name not in PROVIDER_REGISTRY:
        return f"Provider '{provider_name}' not found in registry."
    
    # Initialize provider if needed - pass the session_id
    if not await initialize_provider(provider_name, session_id):
        # Fallback to environment variable failed
        return f"Provider '{provider_name}' initialization failed. Unable to use API key from session or environment."
    
    logger.info(f"Analyzing PDF using model: {model} (provider: {provider_name})")
    
    # Enhanced prompt for PDF analysis
    enhanced_prompt = f"This is a PDF document. Please analyze its content. {prompt}"
    
    # Call the provider's method based on provider type
    try:
        provider = PROVIDER_REGISTRY[provider_name]
        
        if provider_name == "google":
            # For Google's Gemini, we need to convert PDF to images
            try:
                import fitz  # PyMuPDF
                
                # Open the PDF
                pdf_document = fitz.open(pdf_path)
                
                # For simplicity, we'll analyze the first few pages (adjust as needed)
                max_pages = min(5, len(pdf_document))
                
                all_responses = []
                
                for page_num in range(max_pages):
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Render page to an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
                    
                    # Save the image temporarily
                    temp_img_path = f"temp_page_{page_num}.png"
                    pix.save(temp_img_path)
                    
                    # Analyze the image
                    page_prompt = f"{enhanced_prompt} (Page {page_num+1} of {len(pdf_document)})"
                    page_response = await provider.analyze_image(temp_img_path, page_prompt, model, **kwargs)
                    all_responses.append(f"--- Page {page_num+1} Analysis ---\n{page_response}")
                    
                    # Clean up
                    os.remove(temp_img_path)
                
                # Combine responses
                combined_response = "\n\n".join(all_responses)
                
                # If the PDF has more pages than we analyzed, add a note
                if len(pdf_document) > max_pages:
                    combined_response += f"\n\nNote: Only analyzed the first {max_pages} pages of {len(pdf_document)} total pages."
                
                return combined_response
                
            except ImportError:
                logger.warning("PyMuPDF not installed. Falling back to direct PDF handling.")
                # Try direct handling as fallback
                return await provider.analyze_image(pdf_path, enhanced_prompt, model, **kwargs)
        
        elif provider_name == "anthropic" or provider_name == "openai":
            # For Anthropic and OpenAI, we can directly process the PDF in most cases
            # But we need to handle the file differently - read it as bytes
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                
            # Create a base64 encoded version for the API
            pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
            
            # Call a modified version of analyze_image that accepts base64 data
            return await provider.analyze_image(pdf_path, enhanced_prompt, model, 
                                               pdf_data=pdf_data, 
                                               pdf_base64=pdf_base64, 
                                               **kwargs)
        
        else:
            # Generic fallback - try direct processing
            logger.warning(f"No specific PDF handling for provider {provider_name}. Attempting generic approach.")
            return await provider.analyze_image(pdf_path, enhanced_prompt, model, **kwargs)
            
    except Exception as e:
        logger.error(f"Error analyzing PDF with {model}: {str(e)}")
        return f"Error analyzing PDF: {str(e)}"

# Helper function
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Load additional providers when this module is imported
# load_providers_from_directory()

def build_model_registry_from_config():
    """Build MODEL_REGISTRY dictionary from config.json models"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "json", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        registry = {}
        
        # Get models from config
        models = config.get("models", {})
        for model_id, model_info in models.items():
            registry[model_id] = {
                "provider": model_info.get("provider", "").lower(),
                "vision_capable": "image" in model_info.get("capabilities", []),
                "max_tokens": model_info.get("maxTokens", 2048)
            }
        
        return registry
    except Exception as e:
        logger.error(f"Error building model registry from config: {str(e)}")
        # Return default MODEL_REGISTRY if config loading fails
        return MODEL_REGISTRY

# Replace static MODEL_REGISTRY with dynamic loading
MODEL_REGISTRY = build_model_registry_from_config()

