import streamlit as st
import io
import wave
import base64
from langchain_core.messages import HumanMessage
from google.genai import types

def render_text_to_audio(models, model_choice):
    st.header("🔊 Text to Audio - Speech Synthesis")
    text_input = st.text_area("Enter text to convert to speech:", height=150, key="tts_input", placeholder="Enter the text you want to hear...")
    if model_choice == "OpenAI":
        col1, col2 = st.columns(2)
        with col1:
            voice = st.selectbox("Select voice:", ["alloy","echo","fable","onyx","nova","shimmer"])
        with col2:
            speed = st.slider("Speech speed:", 0.25, 4.0, 1.0, 0.25)
        if st.button("🎙️ Generate Audio", use_container_width=True):
            if not text_input.strip() or not models.get("openai_client"):
                st.warning("Enter text and ensure OpenAI client is configured.")
                return
            with st.spinner("Generating audio..."):
                try:
                    response = models["openai_client"].audio.speech.create(model="tts-1-hd", voice=voice, input=text_input, speed=speed)
                    st.audio(response.content, format="audio/mpeg")
                    st.download_button("📥 Download Audio", data=response.content, file_name="speech.mp3", mime="audio/mpeg")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif model_choice == "Gemini":
        col1, col2 = st.columns(2)
        with col1:
            voice_name = st.selectbox("Select voice:", ["Kore","Puck"])
        with col2:
            speed = st.slider("Speech speed:", 0.5, 2.0, 1.0, 0.1)
        if st.button("🎙️ Generate Audio", use_container_width=True):
            if not text_input.strip() or not models.get("gemini_genai_client"):
                st.warning("Enter text and ensure Gemini client is configured.")
                return
            with st.spinner("Generating audio..."):
                try:
                    audio_resp = models["gemini_genai_client"].models.generate_content(
                        model="gemini-3.1-flash-tts-preview",
                        contents=text_input,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                            ))
                        )
                    )
                    parts = audio_resp.candidates[0].content.parts
                    pcm = parts[0].inline_data.data
                    wav_bytes = _pcm_to_wav_bytes(pcm, channels=1, sampwidth=2, framerate=24000)
                    st.audio(wav_bytes, format="audio/wav")
                    st.download_button("📥 Download Audio(WAV)", data=wav_bytes, file_name="gemini_speech.wav", mime="audio/wav")
                except Exception as e:
                    st.error(f"Error: {e}")

def render_audio_to_text(models, model_choice):
    st.header("🎙️ Audio to Text - Transcription")
    audio_file = st.file_uploader("Upload audio file", type=["mp3","wav","m4a","ogg","flac"])
    if not audio_file:
        return
    st.audio(audio_file)
    if st.button("📝 Transcribe Audio", use_container_width=True):
        with st.spinner("Transcribing audio..."):
            try:
                if model_choice == "OpenAI" and models.get("openai_client"):
                    transcript = models["openai_client"].audio.transcriptions.create(model="whisper-1", file=audio_file, language="en")
                    st.info(transcript.text)
                    st.download_button("📋 Copy Transcription", data=transcript.text, file_name="transcription.txt", mime="text/plain")
                elif model_choice == "Gemini" and models.get("gemini"):
                    audio_bytes = audio_file.read()
                    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                    message = HumanMessage(content=[{"type":"text","text":"Transcribe the contents of this audio into English text."},
                                                    {"type":"media","mime_type":"audio/mp3","data":base64_audio}])
                    response = models["gemini"].invoke([message])
                    if isinstance(response.content, list):
                        st.info(response.content[0].get("text",""))
                    else:
                        st.info(response.content)
                else:
                    st.error("Selected model/transcription client is not available.")
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")

def _pcm_to_wav_bytes(pcm_bytes, channels=1, sampwidth=2, framerate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()