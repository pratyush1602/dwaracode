import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("No API key found")
        return
        
    genai.configure(api_key=api_key)
    
    # List available models
    print("Available models:")
    for m in genai.list_models():
        print(m.name)
    
    try:
        # Test with gemini-2.0-flash-exp
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        response = model.generate_content(
            "Hello, how are you?",
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
        print("\nTest response:", response.text)
    except Exception as e:
        print("\nError:", str(e))

if __name__ == "__main__":
    test_gemini() 