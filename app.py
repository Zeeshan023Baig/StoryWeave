import base64
import hashlib
import streamlit as st
from modules.pdf_processor import extract_text_from_pdf
from modules.combined_generator import (
    generate_analysis_and_simplified,
    generate_comic_and_quiz,
    generate_retest_quiz,
    generate_assistant_response,
    get_chapter_summary_and_topics,
)
from modules.image_generator import generate_image_bytes
from modules.audio_generator import generate_audio_bytes
from config.settings import GEMINI_API_KEY, PLAYBACK_SPEED_OPTIONS

st.set_page_config(page_title="StoryWeave", layout="wide", page_icon="📚")

# ── Startup guard ────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    st.error(
        "⚠️ **GEMINI_API_KEY is missing.** "
        "Add it to your `.env` file and restart the app."
    )
    st.stop()

# ── Session-state initialisation ─────────────────────────────────────────────
_STATE_KEYS = {
    "pdf_text": "",
    "pdf_hash": "",          # MD5 of raw PDF bytes
    "gen_level": "",         # learning level used for last generation
    "analysis": None,
    "simplified": None,
    "comic": None,
    "quiz": None,
    "audio_bytes": None,     # raw MP3 bytes (None = not generated / failed)
    "audio_warning": "",     # non-fatal warning message for audio failures
    "audio_speed": 1.0,      # playback rate for explanation audio
    "comic_images": {},      # {panel_index: png_bytes}
    "image_warning": "",     # summary of image-generation failures
    "quiz_submitted": False,
    "user_answers": {},
    "quiz_score": 0,
    # ── Retest ────────────────────────────────────────────────────────────────
    "retest_quiz": None,         # quiz dict for the active retest (None = no retest yet)
    "retest_submitted": False,
    "retest_answers": {},
    "retest_score": 0,
    "all_quiz_questions": [],    # cumulative question objects to exclude across retests
    # ── AI Learning Assistant ─────────────────────────────────────────────────
    "assistant_response": "",
    "assistant_audio_bytes": None,
    "assistant_audio_warning": "",
    "assistant_user_query": "",
}
for k, v in _STATE_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _pdf_hash(raw_bytes: bytes) -> str:
    return hashlib.md5(raw_bytes).hexdigest()


def _cache_hit(current_hash: str, current_level: str) -> bool:
    """True when the same PDF+level was already generated and results are stored."""
    return (
        st.session_state.pdf_hash == current_hash
        and st.session_state.gen_level == current_level
        and st.session_state.analysis is not None
        and st.session_state.comic is not None
    )


def _render_audio_player(audio_bytes: bytes, speed: float = 1.0, player_id: str = "audio-player"):
    """
    Renders an HTML5 audio player that safely sets playbackRate locally.
    Does not use any paid API or modify gTTS fixed speaking rate.
    """
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f"""
    <div style="margin: 6px 0;">
        <audio id="{player_id}" controls style="width: 100%; border-radius: 6px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <script>
            (function() {{
                var a = document.getElementById("{player_id}");
                if (a) {{
                    a.playbackRate = {speed};
                    a.defaultPlaybackRate = {speed};
                }}
            }})();
        </script>
    </div>
    """
    st.components.v1.html(audio_html, height=65)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.title("StoryWeave")
st.subheader("Multimodal Generative AI for Adaptive and Inclusive Learning")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Educational PDF", type=["pdf"])
    learning_level = st.selectbox(
        "Select Learning Level", ["Beginner", "Intermediate", "Advanced"]
    )

    # Automatically extract text from PDF on upload so AI Assistant is ready immediately
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            if len(raw_bytes) > 0:
                current_file_hash = _pdf_hash(raw_bytes)
                if st.session_state.pdf_hash != current_file_hash:
                    uploaded_file.seek(0)
                    pdf_data = extract_text_from_pdf(uploaded_file)
                    if pdf_data and not pdf_data.get("error"):
                        st.session_state.pdf_text = pdf_data.get("text", "")
                        st.session_state.pdf_hash = current_file_hash
                        st.session_state.gen_level = learning_level
                        # Clear old derived data from prior PDF
                        st.session_state.analysis = None
                        st.session_state.simplified = None
                        st.session_state.comic = None
                        st.session_state.quiz = None
                        st.session_state.all_quiz_questions = []
                        st.session_state.assistant_response = ""
                        st.session_state.assistant_audio_bytes = None
                        st.session_state.retest_quiz = None
        except Exception:
            pass

    if st.button("Generate Content", type="primary"):
        # ── Guard: no file uploaded ───────────────────────────────────────────
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
            st.stop()

        # ── Read raw bytes and compute hash ──────────────────────────────────
        raw_bytes = uploaded_file.read()
        if len(raw_bytes) == 0:
            st.error("The uploaded file is empty. Please upload a valid PDF.")
            st.stop()

        current_hash = _pdf_hash(raw_bytes)

        # ── Cache hit — skip Gemini calls ─────────────────────────────────────
        if _cache_hit(current_hash, learning_level):
            st.success("✅ Content already generated for this PDF and level.")
        else:
            # ── PDF extraction ────────────────────────────────────────────────
            with st.spinner("Extracting text from PDF…"):
                try:
                    import io as _io
                    uploaded_file.seek(0)          # rewind after read()
                    pdf_data = extract_text_from_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Could not read PDF: {str(e)}")
                    st.stop()

            if pdf_data is None or pdf_data.get("error"):
                st.error(f"❌ PDF problem: {pdf_data['error'] if pdf_data else 'Unknown error.'}")
                st.stop()

            pdf_text = pdf_data["text"]
            if not pdf_text.strip():
                st.error(
                    "❌ No readable text found in this PDF. "
                    "It may be a scanned/image-only document."
                )
                st.stop()

            # Store new PDF context and reset all derived state
            st.session_state.pdf_text = pdf_text
            st.session_state.pdf_hash = current_hash
            st.session_state.gen_level = learning_level
            st.session_state.analysis = None
            st.session_state.simplified = None
            st.session_state.comic = None
            st.session_state.quiz = None
            st.session_state.audio_bytes = None
            st.session_state.audio_warning = ""
            st.session_state.comic_images = {}
            st.session_state.image_warning = ""
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.session_state.quiz_score = 0
            st.session_state.retest_quiz = None
            st.session_state.retest_submitted = False
            st.session_state.retest_answers = {}
            st.session_state.retest_score = 0
            st.session_state.all_quiz_questions = []

            # ── Gemini Call 1: Analysis + Simplified Text ─────────────────────
            with st.spinner(f"Analysing & generating {learning_level} content…"):
                try:
                    analysis, simplified = generate_analysis_and_simplified(
                        pdf_text, learning_level
                    )
                    st.session_state.analysis = analysis
                    st.session_state.simplified = simplified
                except ValueError as e:
                    st.error(f"❌ Content generation failed (bad response): {str(e)}")
                    st.stop()
                except RuntimeError as e:
                    err = str(e)
                    if "RESOURCE_EXHAUSTED" in err or "429" in err:
                        st.error(
                            "❌ Gemini quota exceeded (429). "
                            "Wait a moment and try again, or check your API plan."
                        )
                    elif "UNAVAILABLE" in err or "503" in err:
                        st.error(
                            "❌ Gemini is temporarily unavailable (503). "
                            "All retries failed. Please try again in a few minutes."
                        )
                    else:
                        st.error(f"❌ Gemini error: {err}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error during analysis: {str(e)}")
                    st.stop()

            # ── Audio (non-fatal) ─────────────────────────────────────────────
            with st.spinner("Generating audio summary…"):
                try:
                    simp = st.session_state.simplified or {}
                    audio_text = (
                        f"Overview. {simp.get('overview', '')} "
                        f"Explanation. {simp.get('detailed_explanation', '')}"
                    ).strip()
                    if audio_text:
                        st.session_state.audio_bytes = generate_audio_bytes(audio_text)
                except RuntimeError as e:
                    st.session_state.audio_warning = str(e)
                except Exception as e:
                    st.session_state.audio_warning = f"Audio generation failed: {str(e)}"

            # ── Gemini Call 2: Comic Script + Quiz ────────────────────────────
            with st.spinner("Generating comic script & quiz…"):
                try:
                    comic, quiz = generate_comic_and_quiz(pdf_text)
                    st.session_state.comic = comic
                    st.session_state.quiz = quiz
                    st.session_state.all_quiz_questions = list(quiz.get("questions", []))
                except ValueError as e:
                    st.error(f"❌ Comic/quiz generation failed (bad response): {str(e)}")
                    st.stop()
                except RuntimeError as e:
                    err = str(e)
                    if "RESOURCE_EXHAUSTED" in err or "429" in err:
                        st.error(
                            "❌ Gemini quota exceeded (429) during comic/quiz generation. "
                            "Analysis content is still available in the other tabs."
                        )
                    elif "UNAVAILABLE" in err or "503" in err:
                        st.error(
                            "❌ Gemini unavailable (503) during comic/quiz generation. "
                            "Analysis content is still available in the other tabs."
                        )
                    else:
                        st.error(f"❌ Comic/quiz error: {err}")
                    # Don't st.stop() — analysis is already saved and displayable

            # ── Comic images (strictly optional, never block script display) ──
            if st.session_state.comic:
                panels = st.session_state.comic.get("panels", [])
                failed_panels = []
                with st.spinner(f"Generating {len(panels)} comic images (optional)…"):
                    for i, panel in enumerate(panels):
                        prompt_text = (
                            panel.get("visual_description")
                            or panel.get("scene_description", "")
                            or panel.get("scene", "")
                        )
                        try:
                            img_bytes = generate_image_bytes(prompt_text)
                            st.session_state.comic_images[i] = img_bytes
                        except RuntimeError as e:
                            failed_panels.append(str(e))
                        except Exception as e:
                            failed_panels.append(f"Panel {i+1}: {str(e)}")

                if failed_panels:
                    # Show ONE consolidated warning, not 6 separate warnings
                    unique_reasons = list(dict.fromkeys(failed_panels))
                    st.session_state.image_warning = unique_reasons[0]

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Ask StoryWeave", "📋 Overview", "📖 Simplified Learning", "🎨 Comic Story", "❓ Quiz"]
)

# ── Tab 0: AI Learning Assistant ──────────────────────────────────────────────
with tab0:
    st.header("💬 Ask StoryWeave — AI Learning Assistant")
    st.caption("Ask custom instructions or questions about your chapter. Grounded strictly in your uploaded text and tailored to your learner level.")

    # Show Summary & Main Topics banner if already analyzed (Zero extra API calls)
    if st.session_state.analysis or st.session_state.simplified:
        chapter_info = get_chapter_summary_and_topics(st.session_state.analysis, st.session_state.simplified)
        if chapter_info["summary"] or chapter_info["main_topics"]:
            with st.expander("📌 Chapter Summary & Key Concepts", expanded=True):
                if chapter_info["summary"]:
                    st.markdown(f"**Concise Summary:** {chapter_info['summary']}")
                if chapter_info["main_topics"]:
                    st.markdown("**Main Topics / Key Concepts:**")
                    st.markdown(" • ".join(f"`{t}`" for t in chapter_info["main_topics"]))

    st.caption(f"🎯 Current Learner Level: **{learning_level}** (adjust in sidebar)")

    has_pdf = bool(st.session_state.pdf_text and st.session_state.pdf_text.strip())
    if not has_pdf:
        st.info("📄 Please upload an educational PDF in the sidebar to ask questions about it.")

    st.markdown("**Quick Prompt Ideas:**")
    eg_col1, eg_col2, eg_col3 = st.columns(3)
    with eg_col1:
        if st.button("📝 Simple Words", key="btn_eg_1"):
            st.session_state["assistant_user_query"] = "Explain this chapter in simple words."
            st.rerun()
        if st.button("🔬 Step by Step", key="btn_eg_2"):
            st.session_state["assistant_user_query"] = "Explain the key process in this chapter step by step."
            st.rerun()
    with eg_col2:
        if st.button("🎯 Exam Key Topics", key="btn_eg_3"):
            st.session_state["assistant_user_query"] = "Give me the most important topics for an exam."
            st.rerun()
        if st.button("📖 Difficult Terms", key="btn_eg_4"):
            st.session_state["assistant_user_query"] = "Explain the difficult terms in this chapter."
            st.rerun()
    with eg_col3:
        if st.button("📝 Revision Notes", key="btn_eg_5"):
            st.session_state["assistant_user_query"] = "Create revision notes from this chapter."
            st.rerun()
        if st.button("⚖️ Compare Concepts", key="btn_eg_6"):
            st.session_state["assistant_user_query"] = "Compare the two main concepts in this chapter."
            st.rerun()

    user_query = st.text_area(
        "Ask StoryWeave about this chapter",
        value=st.session_state.get("assistant_user_query", ""),
        placeholder="e.g. Explain photosynthesis step by step, or give me revision notes...",
        disabled=not has_pdf,
        key="assistant_prompt_input_field",
        height=110,
    )

    if st.button("✨ Generate", type="primary", disabled=not has_pdf, key="btn_generate_assistant_resp"):
        if not user_query.strip():
            st.warning("Please type a question or instruction.")
        else:
            with st.spinner(f"Generating educational response for {learning_level} level…"):
                try:
                    resp = generate_assistant_response(
                        st.session_state.pdf_text,
                        user_query,
                        level=learning_level,
                    )
                    st.session_state.assistant_response = resp
                    st.session_state.assistant_audio_bytes = None
                    st.session_state.assistant_audio_warning = ""
                    st.rerun()
                except (ValueError, RuntimeError) as e:
                    st.error(f"❌ AI Assistant error: {e}")

    # Display generated response clearly below the input
    if st.session_state.assistant_response:
        st.subheader("💡 Adapted Explanation")
        st.markdown(st.session_state.assistant_response)

        # Audio section for the assistant response
        st.write("---")
        st.subheader("🔊 Audio Version")
        col_audio_btn, col_audio_speed = st.columns([2, 1])
        with col_audio_speed:
            asst_speed = st.selectbox(
                "Playback Speed",
                PLAYBACK_SPEED_OPTIONS,
                index=PLAYBACK_SPEED_OPTIONS.index(1.0),
                format_func=lambda s: f"{s}x",
                key="asst_speed_selector",
            )
        with col_audio_btn:
            if st.button("🎧 Generate Audio for Response", key="btn_asst_audio_gen"):
                with st.spinner("Generating audio…"):
                    try:
                        st.session_state.assistant_audio_bytes = generate_audio_bytes(
                            st.session_state.assistant_response
                        )
                        st.session_state.assistant_audio_warning = ""
                        st.rerun()
                    except Exception as e:
                        st.session_state.assistant_audio_warning = str(e)

        if st.session_state.assistant_audio_bytes:
            _render_audio_player(
                st.session_state.assistant_audio_bytes,
                speed=asst_speed,
                player_id="player-asst-audio",
            )
        elif st.session_state.assistant_audio_warning:
            st.warning(f"⚠️ Audio unavailable: {st.session_state.assistant_audio_warning}")

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        st.header(analysis.get("chapter_title", "Chapter Overview"))
        st.write(f"**Topic:** {analysis.get('topic', 'N/A')}")
        st.write(f"**Difficulty:** {analysis.get('difficulty', 'N/A')}")

        # Summary if available
        sum_info = get_chapter_summary_and_topics(analysis, st.session_state.simplified)
        if sum_info["summary"]:
            st.info(f"**Summary:** {sum_info['summary']}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Concepts")
            for concept in analysis.get("key_concepts", []):
                st.markdown(f"- {concept}")
            st.subheader("Key Terms")
            for term in analysis.get("key_terms", []):
                st.markdown(f"- {term}")
        with col2:
            st.subheader("Important Facts")
            for fact in analysis.get("important_facts", []):
                st.markdown(f"- {fact}")
            st.subheader("Learning Objectives")
            for obj in analysis.get("learning_objectives", []):
                st.markdown(f"- {obj}")
            st.subheader("Prerequisites")
            for pre in analysis.get("prerequisite_concepts", []):
                st.markdown(f"- {pre}")
    else:
        st.info("Upload a PDF and click **Generate Content** to see the overview.")

# ── Tab 2: Simplified Learning ────────────────────────────────────────────────
with tab2:
    if st.session_state.simplified:
        simplified = st.session_state.simplified
        st.header("Adaptive Learning Content")

        # Audio player with speed control
        st.subheader("🔊 Listen to this explanation")
        if st.session_state.audio_bytes:
            aud_col, spd_col = st.columns([3, 1])
            with spd_col:
                simp_speed = st.selectbox(
                    "Playback Speed",
                    PLAYBACK_SPEED_OPTIONS,
                    index=PLAYBACK_SPEED_OPTIONS.index(1.0),
                    format_func=lambda s: f"{s}x",
                    key="simp_speed_selector",
                )
            with aud_col:
                _render_audio_player(
                    st.session_state.audio_bytes,
                    speed=simp_speed,
                    player_id="player-simp-audio",
                )
        elif st.session_state.audio_warning:
            st.warning(
                f"⚠️ Audio unavailable: {st.session_state.audio_warning} "
                "(requires an internet connection to Google TTS)"
            )
        else:
            st.caption("Audio not yet generated.")

        st.write(f"**Overview:** {simplified.get('overview', '')}")
        st.subheader("Detailed Explanation")
        st.write(simplified.get("detailed_explanation", ""))
        st.subheader("Key Concepts")
        for kc in simplified.get("key_concepts", []):
            if isinstance(kc, dict):
                st.markdown(f"**{kc.get('concept', '')}**: {kc.get('explanation', '')}")
            else:
                st.markdown(f"- {kc}")
        st.subheader("Important Terms")
        for term in simplified.get("important_terms", []):
            if isinstance(term, dict):
                st.markdown(f"**{term.get('term', '')}**: {term.get('definition', '')}")
            else:
                st.markdown(f"- {term}")
        st.subheader("Quick Summary")
        st.write(simplified.get("quick_summary", ""))
    else:
        st.info("Generate content to see simplified learning materials.")

# ── Tab 3: Comic Story ────────────────────────────────────────────────────────
with tab3:
    if st.session_state.comic:
        comic = st.session_state.comic
        st.header(comic.get("comic_title", "Educational Comic"))

        # Show consolidated image warning once at the top (not per-panel)
        if st.session_state.image_warning:
            st.warning(
                f"⚠️ **Comic images unavailable:** {st.session_state.image_warning}  \n"
                "The full comic script is shown below."
            )

        panels = comic.get("panels", [])
        if not panels:
            st.warning("No comic panels were generated.")
        else:
            for i, panel in enumerate(panels):
                panel_num = panel.get("panel_number", i + 1)
                with st.expander(f"Panel {panel_num}", expanded=True):
                    # Image — shown only if available; script always shown
                    if i in st.session_state.comic_images:
                        st.image(
                            st.session_state.comic_images[i],
                            use_container_width=True,
                        )

                    # Script — always rendered regardless of image status
                    scene = (
                        panel.get("scene_description")
                        or panel.get("scene")
                        or ""
                    )
                    st.write(f"**Scene:** {scene}")

                    characters = panel.get("characters", [])
                    if isinstance(characters, list):
                        st.write(f"**Characters:** {', '.join(characters)}")
                    elif characters:
                        st.write(f"**Characters:** {characters}")

                    narration = panel.get("narration", "")
                    if narration:
                        st.write(f"**Narration:** {narration}")

                    dialogue = panel.get("dialogue", "")
                    if dialogue:
                        st.write(f"**Dialogue:** {dialogue}")
    else:
        st.info("Generate content to see the comic script.")

# ── Tab 4: Quiz ───────────────────────────────────────────────────────────────
with tab4:
    if st.session_state.quiz:
        # ── Determine which quiz is active ────────────────────────────────────
        # After a retest is generated we show the retest; otherwise the original.
        active_quiz    = st.session_state.retest_quiz or st.session_state.quiz
        is_retest      = st.session_state.retest_quiz is not None
        form_key       = "retest_form" if is_retest else "quiz_form"
        answers_key    = "retest_answers" if is_retest else "user_answers"
        submitted_key  = "retest_submitted" if is_retest else "quiz_submitted"
        score_key      = "retest_score" if is_retest else "quiz_score"

        quiz_label = active_quiz.get("quiz_title", "Chapter Quiz")
        if is_retest:
            st.header(f"🔄 {quiz_label}")
            st.caption("This is a new set of questions — different from your previous attempts.")
        else:
            st.header(quiz_label)

        questions = active_quiz.get("questions", [])

        if not st.session_state[submitted_key]:
            with st.form(form_key):
                for i, q in enumerate(questions):
                    st.markdown(f"**Q{i+1}: {q.get('question', '')}**")
                    st.session_state[answers_key][i] = st.radio(
                        "Select your answer:",
                        q.get("options", []),
                        key=f"{form_key}_q_{i}",
                        index=None,
                    )
                    st.write("---")

                if st.form_submit_button("Submit Quiz"):
                    st.session_state[submitted_key] = True
                    score = sum(
                        1
                        for i, q in enumerate(questions)
                        if st.session_state[answers_key].get(i) == q.get("correct_answer")
                    )
                    st.session_state[score_key] = score
                    st.rerun()

        if st.session_state[submitted_key]:
            total = len(questions)
            score = st.session_state[score_key]
            pct   = (score / total * 100) if total > 0 else 0
            st.success(f"### Score: {score} / {total}  ({pct:.0f}%)")

            st.subheader("Explanations")
            for i, q in enumerate(questions):
                user_ans   = st.session_state[answers_key].get(i)
                correct_ans = q.get("correct_answer")
                with st.expander(f"Q{i+1}: {q.get('question', '')}"):
                    st.write(f"**Your Answer:** {user_ans}")
                    st.write(f"**Correct Answer:** {correct_ans}")
                    if user_ans == correct_ans:
                        st.success("✅ Correct!")
                    else:
                        st.error("❌ Incorrect.")
                    st.info(f"**Explanation:** {q.get('explanation', '')}")

            # ── Retest button — only shown after original quiz is submitted ────
            st.divider()
            if st.button("🔄 Retest — Generate New Questions", type="primary"):
                # Use accumulated exclusions across all retests
                prev_questions = list(st.session_state.all_quiz_questions)
                if not prev_questions and st.session_state.quiz:
                    prev_questions = st.session_state.quiz.get("questions", [])

                with st.spinner("Generating a new set of questions…"):
                    try:
                        new_quiz = generate_retest_quiz(
                            st.session_state.pdf_text,
                            previous_questions=prev_questions,
                            level=st.session_state.gen_level or learning_level,
                        )
                        # Accumulate new questions into all_quiz_questions for subsequent retests
                        for q in new_quiz.get("questions", []):
                            if q not in st.session_state.all_quiz_questions:
                                st.session_state.all_quiz_questions.append(q)

                        st.session_state.retest_quiz      = new_quiz
                        st.session_state.retest_submitted = False
                        st.session_state.retest_answers   = {}
                        st.session_state.retest_score     = 0
                        st.rerun()
                    except (ValueError, RuntimeError) as e:
                        st.error(f"❌ Could not generate retest: {e}")
    else:
        st.info("Generate content to take the quiz.")



