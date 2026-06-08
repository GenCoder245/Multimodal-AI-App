import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 Multimodal AI Applications")
st.markdown("*Transform content across different modalities using AI models*")

# Initialize Models
@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        openai_model = ChatOpenAI(model="gpt-4o-mini")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return gemini_model, openai_model, openai_client
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

gemini_model, openai_model, openai_client = load_models()

# Sidebar Navigation
st.sidebar.header("🔧 Configuration")
modality_type = st.sidebar.selectbox(
    "Select Conversion Type:",
    [
        "Text to Text",
        "Text to Image",
        "Image to Text",
        "Text to Audio",
        "Audio to Text"
    ],
    help="Choose the type of modality conversion you want to perform"
)

model_choice = st.sidebar.selectbox(
    "Select Model Provider:",
    ["Gemini", "OpenAI"],
    help="Choose which AI provider to use"
)

# ============ TEXT TO TEXT ============
if modality_type == "Text to Text":
    st.header("📝 Text to Text - Translation & Transformation")
    
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.selectbox(
            "Source Language:",
            ["English"],
            key="source_lang"
        )
    
    with col2:
        target_language = st.selectbox(
            "Target Language:",
            ["Spanish", "French", "German", "Hindi", "Tamil", "Japanese", "Chinese"],
            key="target_lang"
        )
    
    user_text = st.text_area(
        "Enter text to translate:",
        height=150,
        placeholder="Enter the text you want to translate...",
        key="text_input"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Translate", use_container_width=True):
            if user_text.strip():
                with st.spinner("Translating..."):
                    try:
                        model = gemini_model if model_choice == "Gemini" else openai_model
                        
                        prompt = f"Translate the following {source_language} text to {target_language}. Only provide the translation, no explanations:\n\n{user_text}"
                        
                        response = model.invoke([HumanMessage(content=prompt)])
                        
                        st.success("✓ Translation Complete")
                        st.info(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text to translate")

# ============ TEXT TO IMAGE ============
elif modality_type == "Text to Image":
    st.header("🖼️ Text to Image - Generation")
    
    prompt = st.text_area(
        "Describe the image you want to generate:",
        height=150,
        placeholder="Example: A serene landscape with mountains and a lake at sunset...",
        key="image_prompt"
    )
    
    quality = st.selectbox(
        "Image Quality:",
        ["low", "high"],
        help="Lower quality generates faster"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎨 Generate Image", use_container_width=True):
            if prompt.strip():
                with st.spinner("Generating image..."):
                    try:
                        if model_choice == "OpenAI" and openai_model:
                            # Using OpenAI's image generation
                            response = openai_model.invoke(
                                [HumanMessage(content=prompt)],
                                tools=[{"type": "image_generation", "quality": quality}],
                                tool_choice={"type": "image_generation"}
                            )
                            
                            # Extract and display the image
                            for block in response.content_blocks:
                                if block["type"] == "image":
                                    st.success("✓ Image Generated")
                                    image_base64 = block["base64"]
                                    st.image(
                                        base64.b64decode(image_base64),
                                        caption="Generated Image",
                                        use_container_width=True
                                    )
                        else:
                            st.info("Text-to-Image generation is available with OpenAI model. Please select OpenAI from the sidebar.")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
            else:
                st.warning("Please describe the image you want to generate")

# ============ IMAGE TO TEXT ============
elif modality_type == "Image to Text":
    st.header("🔍 Image to Text - Analysis")
    
    image_source = st.radio(
        "Choose image source:",
        ["📤 Upload Image", "🔗 Image URL"],
        horizontal=True
    )
    
    if image_source == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            help="Supported formats: JPG, PNG, GIF, WebP"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            image_bytes = uploaded_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            analysis_type = st.selectbox(
                "Type of Analysis:",
                ["General Description", "Detailed Analysis", "Extract Text (OCR)", "Scene Understanding"]
            )
            
            prompts = {
                "General Description": "Describe this image in a few sentences.",
                "Detailed Analysis": "Provide a detailed analysis of this image including all elements, colors, composition, and mood.",
                "Extract Text (OCR)": "Extract and transcribe all text visible in this image.",
                "Scene Understanding": "Analyze the scene in this image. What's happening? What are the main elements?"
            }
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        model = gemini_model if model_choice == "Gemini" else openai_model
                        
                        message = HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": prompts.get(analysis_type, "Describe this image.")
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{base64_image}"
                                }
                            ]
                        )
                        
                        response = model.invoke([message])
                        st.success("✓ Analysis Complete")
                        st.info(response.content)
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
    
    else:  # Image URL
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/image.jpg"
        )
        
        if image_url:
            st.image(image_url, caption="Image from URL", use_container_width=True)
            
            analysis_type = st.selectbox(
                "Type of Analysis:",
                ["General Description", "Detailed Analysis", "Extract Text (OCR)", "Scene Understanding"]
            )
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        model = gemini_model if model_choice == "Gemini" else openai_model
                        
                        prompts = {
                            "General Description": "Describe this image in a few sentences.",
                            "Detailed Analysis": "Provide a detailed analysis of this image.",
                            "Extract Text (OCR)": "Extract all text visible in this image.",
                            "Scene Understanding": "Analyze what's happening in this scene."
                        }
                        
                        message = HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": prompts.get(analysis_type, "Describe this image.")
                                },
                                {
                                    "type": "image_url",
                                    "image_url": image_url
                                }
                            ]
                        )
                        
                        response = model.invoke([message])
                        st.success("✓ Analysis Complete")
                        st.info(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============ TEXT TO AUDIO ============
elif modality_type == "Text to Audio":
    st.header("🔊 Text to Audio - Speech Synthesis")
    
    text_input = st.text_area(
        "Enter text to convert to speech:",
        height=150,
        placeholder="Enter the text you want to hear...",
        key="tts_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        voice = st.selectbox(
            "Select voice:",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            help="Different voices with different characteristics"
        )
    
    with col2:
        speed = st.slider(
            "Speech speed:",
            min_value=0.25,
            max_value=4.0,
            value=1.0,
            step=0.25
        )
    
    if st.button("🎙️ Generate Audio", use_container_width=True):
        if text_input.strip() and openai_client:
            with st.spinner("Generating audio..."):
                try:
                    response = openai_client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=text_input,
                        speed=speed
                    )
                    
                    st.success("✓ Audio Generated")
                    st.audio(response.content, format="audio/mpeg")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Audio",
                        data=response.content,
                        file_name="generated_speech.mp3",
                        mime="audio/mpeg"
                    )
                except Exception as e:
                    st.error(f"Error generating audio: {e}")
        else:
            st.warning("Please enter some text to convert to speech")

# ============ AUDIO TO TEXT ============
elif modality_type == "Audio to Text":
    st.header("🎙️ Audio to Text - Transcription")
    
    audio_source = st.radio(
        "Choose audio source:",
        # ["📤 Upload Audio File", "📹 Record Audio"],
        # At this point only audio file upload is supported.
        ["📤 Upload Audio File"],
        horizontal=True
    )
    
    if audio_source == "📤 Upload Audio File":
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
        )
        
        if audio_file and openai_client:
            st.audio(audio_file)
            
            if st.button("📝 Transcribe Audio", use_container_width=True):
                with st.spinner("Transcribing audio..."):
                    try:
                        transcript = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="en"
                        )
                        
                        st.success("✓ Transcription Complete")
                        st.info(transcript.text)
                        
                        # Copy to clipboard button
                        st.download_button(
                            label="📋 Copy Transcription",
                            data=transcript.text,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error transcribing audio: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Multimodal AI Assistant** v1.0
    
    Built with:
    - 🦜 LangChain
    - 🤖 Google Gemini & OpenAI
    - 📊 Streamlit
    
    **Features:**
    - Text Translation
    - Image Generation & Analysis
    - Speech Synthesis
    - Audio Transcription
    """
)

# Display API Status
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
if gemini_model and openai_model:
    st.sidebar.success("✓ All models loaded successfully")
else:
    st.sidebar.warning("⚠️ Some models failed to load. Check API keys.")
