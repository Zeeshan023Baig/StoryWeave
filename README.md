# StoryWeave

A Multimodal Generative AI Framework for Adaptive and Inclusive Learning.

## Project Overview
StoryWeave takes an educational PDF chapter and automatically generates adaptive learning materials including chapter analysis, simplified adaptive text for different learning levels, an educational comic script, and an interactive quiz.

## Architecture
- **Streamlit**: Web interface and interactive quiz.
- **PyMuPDF**: Robust PDF parsing and text extraction.
- **Google Gemini**: LLM for content analysis, simplification, and generation.

## Folder Structure
```
StoryWeave/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── config/
│   └── settings.py          # Configuration and environment loading
├── modules/
│   ├── pdf_processor.py     # PyMuPDF text extraction
│   ├── llm_service.py       # Gemini API integration
│   ├── content_analyzer.py  # Chapter analysis
│   ├── simplified_text.py   # Adaptive text generation
│   ├── comic_generator.py   # 6-panel comic script generation
│   └── quiz_generator.py    # 5-question MCQ generation
├── prompts/                 # Text files containing LLM prompts
└── utils/                   # Helper functions
```

## Installation
1. Clone the repository or navigate to the project directory.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Gemini API Setup
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your `GEMINI_API_KEY`.

## Running the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

## Current Features (MVP Phase 1)
- PDF Text Extraction (PyMuPDF)
- Chapter Analysis (Topic, concepts, facts, objectives)
- Adaptive Simplified Learning (Beginner, Intermediate, Advanced)
- 6-Panel Comic Script Generation
- Interactive 5-Question MCQ Quiz with Explanations

## Future Features (Phase 2 & 3)
- AI Comic Image Generation
- Text-to-Speech (TTS)
- User Authentication
- Database and Analytics
- Content Recommendation System
