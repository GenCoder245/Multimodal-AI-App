import sys
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.models import load_models
from src.chat import render_chat
from src.image import render_text_to_image, render_image_to_text
from src.audio import render_text_to_audio, render_audio_to_text

st.set_page_config(page_title="Multimodal AI Assistant", page_icon="🎨", layout="wide")

st.title("🎨 Multimodal AI Applications")
st.markdown("*Transform content across different modalities using AI models*")

@st.cache_resource
def _load():
    return load_models()

models = _load()

st.sidebar.header("🔧 Configuration")
modality_type = st.sidebar.selectbox(
    "Select Conversion Type:",
    ["Text to Text", "Text to Image", "Image to Text", "Text to Audio", "Audio to Text"]
)

if modality_type == "Text to Text":
    available_models = ["OpenAI", "Gemini", "Groq", "HuggingFace", "Ollama"]
elif modality_type == "Text to Image":
    available_models = ["OpenAI", "Gemini", "Groq"]
elif modality_type == "Image to Text":
    available_models = ["OpenAI", "Gemini", "Groq"]
elif modality_type == "Text to Audio":
    available_models = ["OpenAI", "Gemini"]
else:
    available_models = ["OpenAI", "Gemini"]

model_choice = st.sidebar.selectbox("Select Model Provider:", available_models)

if modality_type == "Text to Text":
    render_chat(models, model_choice)
elif modality_type == "Text to Image":
    render_text_to_image(models, model_choice)
elif modality_type == "Image to Text":
    render_image_to_text(models, model_choice)
elif modality_type == "Text to Audio":
    render_text_to_audio(models, model_choice)
elif modality_type == "Audio to Text":
    render_audio_to_text(models, model_choice)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Multimodal AI Assistant** v2.0
    
    Built with:
    - 🦜 LangChain
    - 🤖 OpenAI, Gemini, Groq, Ollama, HuggingFace
    - 📊 Streamlit
    
    **Features:**
    - AI Chatbot (Text to Text)
    - Image Generation & Analysis
    - Speech Synthesis & Transcription
    - Multiple Model Providers
    - Local Models Support (Ollama)
    """
)

# Display API Status
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
# st.sidebar.info(
#     "Multimodal AI Assistant — Streamlit frontend, multiple providers (OpenAI, Gemini, Groq, Ollama, HF)."
# )
if models:
    st.sidebar.success("✓ All models loaded successfully")
else:
    st.sidebar.warning("⚠️ Some models failed to load. Check API keys.")