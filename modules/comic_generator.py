import os
from modules.llm_service import generate_json

def generate_comic_script(text):
    """
    Generates a 6-panel comic script based on the text.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'comic_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{text}", text)
    
    return generate_json(prompt, system_instruction="You are a creative educational comic writer.")
