import streamlit as st
from modules.pdf_processor import extract_text_from_pdf
from modules.content_analyzer import analyze_chapter_content
from modules.simplified_text import generate_simplified_content
from modules.comic_generator import generate_comic_script
from modules.quiz_generator import generate_quiz

st.set_page_config(page_title="StoryWeave", layout="wide", page_icon="📚")

# Initialize session state variables
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "simplified" not in st.session_state:
    st.session_state.simplified = None
if "comic" not in st.session_state:
    st.session_state.comic = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

st.title("StoryWeave")
st.subheader("Multimodal Generative AI for Adaptive and Inclusive Learning")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Educational PDF", type=["pdf"])
    learning_level = st.selectbox("Select Learning Level", ["Beginner", "Intermediate", "Advanced"])
    
    if st.button("Generate Content", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Extracting text from PDF..."):
                pdf_data = extract_text_from_pdf(uploaded_file)
                
            if pdf_data["error"]:
                st.error(pdf_data["error"])
            else:
                st.session_state.pdf_text = pdf_data["text"]
                
                with st.spinner("Analyzing Chapter Content..."):
                    try:
                        st.session_state.analysis = analyze_chapter_content(st.session_state.pdf_text)
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        
                with st.spinner(f"Generating {learning_level} Content..."):
                    try:
                        st.session_state.simplified = generate_simplified_content(st.session_state.pdf_text, learning_level)
                    except Exception as e:
                        st.error(f"Simplified content generation failed: {str(e)}")
                        
                with st.spinner("Generating Comic Script..."):
                    try:
                        st.session_state.comic = generate_comic_script(st.session_state.pdf_text)
                    except Exception as e:
                        st.error(f"Comic script generation failed: {str(e)}")
                        
                with st.spinner("Generating Quiz..."):
                    try:
                        st.session_state.quiz = generate_quiz(st.session_state.pdf_text)
                        # Reset quiz state
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}
                        st.session_state.quiz_score = 0
                    except Exception as e:
                        st.error(f"Quiz generation failed: {str(e)}")

# UI Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Simplified Learning", "Comic Story", "Quiz"])

with tab1:
    if st.session_state.analysis:
        st.header(st.session_state.analysis.get("chapter_title", "Chapter Overview"))
        st.write(f"**Topic:** {st.session_state.analysis.get('topic', 'N/A')}")
        st.write(f"**Difficulty:** {st.session_state.analysis.get('difficulty', 'N/A')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Concepts")
            for concept in st.session_state.analysis.get("key_concepts", []):
                st.markdown(f"- {concept}")
                
            st.subheader("Key Terms")
            for term in st.session_state.analysis.get("key_terms", []):
                st.markdown(f"- {term}")
                
        with col2:
            st.subheader("Important Facts")
            for fact in st.session_state.analysis.get("important_facts", []):
                st.markdown(f"- {fact}")
                
            st.subheader("Learning Objectives")
            for obj in st.session_state.analysis.get("learning_objectives", []):
                st.markdown(f"- {obj}")
                
            st.subheader("Prerequisites")
            for pre in st.session_state.analysis.get("prerequisite_concepts", []):
                st.markdown(f"- {pre}")
    else:
        st.info("Upload a PDF and click 'Generate Content' to see the overview.")

with tab2:
    if st.session_state.simplified:
        st.header("Adaptive Learning Content")
        st.write(f"**Overview:** {st.session_state.simplified.get('overview', '')}")
        
        st.subheader("Detailed Explanation")
        st.write(st.session_state.simplified.get('detailed_explanation', ''))
        
        st.subheader("Key Concepts")
        for kc in st.session_state.simplified.get("key_concepts", []):
            st.markdown(f"**{kc.get('concept', '')}**: {kc.get('explanation', '')}")
            
        st.subheader("Important Terms")
        for term in st.session_state.simplified.get("important_terms", []):
            st.markdown(f"**{term.get('term', '')}**: {term.get('definition', '')}")
            
        st.subheader("Quick Summary")
        st.write(st.session_state.simplified.get('quick_summary', ''))
    else:
        st.info("Generate content to see simplified learning materials.")

with tab3:
    if st.session_state.comic:
        st.header(st.session_state.comic.get("comic_title", "Educational Comic"))
        st.info("AI Comic Images — Coming in Phase 2")
        
        for panel in st.session_state.comic.get("panels", []):
            with st.expander(f"Panel {panel.get('panel_number', '')}", expanded=True):
                st.write(f"**Scene:** {panel.get('scene_description', '')}")
                
                # Check if characters is a list before joining
                characters = panel.get('characters', [])
                if isinstance(characters, list):
                    st.write(f"**Characters:** {', '.join(characters)}")
                else:
                    st.write(f"**Characters:** {characters}")
                    
                st.write(f"**Narration:** {panel.get('narration', '')}")
                st.write(f"**Dialogue:** {panel.get('dialogue', '')}")
    else:
        st.info("Generate content to see the comic script.")

with tab4:
    if st.session_state.quiz:
        st.header(st.session_state.quiz.get("quiz_title", "Chapter Quiz"))
        
        questions = st.session_state.quiz.get("questions", [])
        
        # Build the form
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}: {q.get('question', '')}**")
                # Using radio buttons
                st.session_state.user_answers[i] = st.radio(
                    "Select your answer:",
                    q.get("options", []),
                    key=f"q_{i}",
                    index=None
                )
                st.write("---")
            
            submit_button = st.form_submit_button("Submit Quiz")
            
            if submit_button:
                st.session_state.quiz_submitted = True
                
                # Calculate score
                score = 0
                for i, q in enumerate(questions):
                    if st.session_state.user_answers.get(i) == q.get("correct_answer"):
                        score += 1
                st.session_state.quiz_score = score

        # Show results outside the form if submitted
        if st.session_state.quiz_submitted:
            st.success(f"### Score: {st.session_state.quiz_score} / {len(questions)}")
            percentage = (st.session_state.quiz_score / len(questions)) * 100 if len(questions) > 0 else 0
            st.write(f"**Percentage: {percentage:.0f}%**")
            
            st.subheader("Explanations")
            for i, q in enumerate(questions):
                user_ans = st.session_state.user_answers.get(i)
                correct_ans = q.get("correct_answer")
                
                with st.expander(f"Q{i+1}: {q.get('question', '')}"):
                    st.write(f"**Your Answer:** {user_ans}")
                    st.write(f"**Correct Answer:** {correct_ans}")
                    
                    if user_ans == correct_ans:
                        st.success("Correct!")
                    else:
                        st.error("Incorrect.")
                        
                    st.info(f"**Explanation:** {q.get('explanation', '')}")
    else:
        st.info("Generate content to take the quiz.")
