import os
from modules.llm_service import generate_json

def generate_simplified_content(text, level):
    """
    Generates simplified adaptive content based on the learning level.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'simplified_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{level}", level).replace("{text}", text)
    
    return generate_json(prompt, system_instruction="You are a helpful educational AI assistant.")
