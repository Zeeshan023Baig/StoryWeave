import io
from gtts import gTTS


def generate_audio_bytes(text):
    """
    Generates an MP3 audio clip from text using gTTS and returns raw bytes.

    Returns bytes suitable for st.audio(data, format='audio/mp3').
    Raises RuntimeError on any failure so the caller can show a friendly warning.
    """
    if not text or not text.strip():
        raise RuntimeError("Cannot generate audio: no text provided.")
    try:
        tts = gTTS(text=text.strip(), lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {str(e)}")


from config.settings import PLAYBACK_SPEED_OPTIONS

# Backwards-compatible alias used by older callers
def generate_audio_file(text):
    """Deprecated: use generate_audio_bytes() instead. Kept for compatibility."""
    return generate_audio_bytes(text)

