from google import genai
from google.genai import types
import json
from config.settings import GEMINI_API_KEY, MODEL_NAME

def get_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please configure it in .env.")
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_text(prompt, system_instruction=None):
    client = get_client()
    try:
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")

def generate_json(prompt, system_instruction=None):
    client = get_client()
    try:
        config_kwargs = {"response_mime_type": "application/json"}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        text_response = response.text
        return json.loads(text_response.strip())
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from Gemini response.")
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")
