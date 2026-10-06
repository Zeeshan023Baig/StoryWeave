import os
from modules.llm_service import generate_json

def analyze_chapter_content(text):
    """
    Analyzes the extracted text to gather chapter details.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'analysis_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{text}", text)
    
    return generate_json(prompt, system_instruction="You are a helpful educational AI assistant.")
