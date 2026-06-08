import streamlit as st
import os
import base64
from dotenv import load_dotenv
from pathlib import Path
# For audio processing
import wave
import io

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from openai import OpenAI
from google import genai
from google.genai import types

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
            model="gemini-2.5-flash-lite",  #gemini-2.5-flash
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        openai_model = ChatOpenAI(model="gpt-5.4-nano")  #   gpt-4o-mini
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # llama-3.3-70b-versatile, qwen/qwen3-32b
        # "meta-llama/llama-4-scout-17b-16e-instruct" models supports multimodality,
        #  while I previously used "llama-3.3-70b-versatile" which is a text-only model. 

        groq_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=os.getenv("GROQ_API_KEY"))
        
        # Ollama model
        ollama_model = ChatOllama(model="gemma3:1b", temperature=0.7)
        
        # Google Genai client for TTS
        gemini_genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        return {
            "gemini": gemini_model,
            "openai": openai_model,
            "openai_client": openai_client,
            "groq": groq_model,
            "ollama": ollama_model,
            "gemini_genai_client": gemini_genai_client
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_models()

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

# Determine available models based on modality
if modality_type == "Text to Text":
    available_models = ["OpenAI", "Gemini", "Groq", "HuggingFace", "Ollama"]
elif modality_type == "Text to Image":  # Only OpenAI works
    available_models = ["OpenAI", "Gemini", "Groq"]
elif modality_type == "Image to Text":
    available_models = ["OpenAI", "Gemini", "Groq"] # , "HuggingFace"]
elif modality_type == "Text to Audio":
    available_models = ["OpenAI", "Gemini"]
else:  # Audio to Text
    available_models = ["OpenAI", "Gemini"]

model_choice = st.sidebar.selectbox(
    "Select Model Provider:",
    available_models,
    help="Choose which AI provider to use"
)

# ============ TEXT TO TEXT - CHATBOT ============
# All good - OpenAI, Gemini, Groq, HuggingFace, Ollama
if modality_type == "Text to Text":
    st.header("💬 Text to Text - AI Chatbot")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.spinner("Generating response..."):
            try:
                # Select model
                if model_choice == "Gemini":
                    model = models["gemini"]
                elif model_choice == "OpenAI":
                    model = models["openai"]
                elif model_choice == "Groq":
                    model = models["groq"]
                elif model_choice == "Ollama":
                    model = models["ollama"]
                elif model_choice == "HuggingFace":
                    # HuggingFace setup
                    hf_llm = HuggingFaceEndpoint(
                        repo_id="deepseek-ai/DeepSeek-V4-Flash",
                        task="text-generation",
                        max_new_tokens=1024,
                        huggingfacehub_api_token=os.getenv("HF_API_TOKEN")
                    )
                    model = ChatHuggingFace(llm=hf_llm)
                
                # Get response from model
                response = model.invoke(st.session_state.chat_history)
                
                # Add assistant response to history
                if model_choice == "Gemini":  # Only for Gemini3 Models, output comes as list
                    if isinstance(response.content, list):
                        response_content = response.content[0]['text']
                        st.session_state.chat_history.append(AIMessage(content=response_content))
                    else:
                        st.session_state.chat_history.append(AIMessage(content=response.content))
                else:
                    st.session_state.chat_history.append(AIMessage(content=response.content))
                
                # Rerun to display new message
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Clear chat button
    col1, col2, col3 = st.columns(3)

    
    # col1, col2, col3 = st.columns(3)   is not giving any error, but 
    # col1 = st.columns(1) giving errors saying TypeError: 'list' object does not support the context manager protocol

    #col1 = st.columns(1)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ============ TEXT TO IMAGE ============
# OpenAI works. Gemini pending -  Unable to verify since Gemini quota exceeded. Will check once quota is back (Jun-8).
# Groq no feature.
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
        ["low","medium","high"],
        help="Lower quality generates faster"
    )
    
    col1, col2, col3 = st.columns(3)
    # Need to try "Text to Image in Gemini"
    # Gemini Text-to-Image generation is available through Google AI Studio
    with col1:
        if st.button("🎨 Generate Image", use_container_width=True):
            if prompt.strip():
                with st.spinner("Generating image..."):
                    try:
                        if model_choice == "OpenAI":
                            # Using OpenAI's image generation
                            response = models["openai"].invoke(
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
                        elif model_choice == "Gemini":
                            # Using Gemini's image generation via LangChain
                            try: # This is generating text response instead of image.
                                # Need to check further.
                                # Might need to create a separate gemini model using ChatGoogleGenerativeAI with model="gemini-3.1-flash-image" which is more stable for image generation.
                                
                                # response = models["gemini"].invoke(
                                #     [HumanMessage(content=prompt)]
                                # )
                                
                                # # Extract the image data from response
                                # if hasattr(response, 'content'):
                                #     response_content = response.content
                                    
                                #     # Check if response contains image data
                                #     if isinstance(response_content, str):
                                #         st.success("✓ Image Generated (Response received)")
                                #         st.info(f"Image generation response: {response_content}")
                                #     else:
                                #         st.success("✓ Image Generated")
                                #         st.write(response_content)
                                # else:
                                #     st.info("Gemini Text-to-Image generation requires explicit image generation capability. Using native SDK approach...")
                                    
                                    # Fallback to native Gemini SDK for image generation
                                    # from google.generativeai import upload_file
                                    # import io

                                    from google import genai
                                    from google.genai import types
                                    from io import BytesIO
                                    
                                    # image_response = models["gemini_genai_client"].models.generate_images(
                                    #     model="imagen-3.0-generate-002",
                                    #     prompt=prompt,
                                    #     number_of_images=1,
                                    #     safety_filter_level="block_none",
                                    #     person_generation="allow_adult",
                                    # )
                                    
                                    # if image_response.images:
                                    #     st.success("✓ Image Generated")
                                    #     # Display the generated image
                                    #     image_data = image_response.images[0]._image_bytes
                                    #     st.image(image_data, caption="Generated Image", use_container_width=True)

                                    # Gemini Quota exceeded. So unable to check "Image to Text" as of now (Jun-8)
                                    client = models["gemini_genai_client"]
                                    response = client.models.generate_content(
                                        model="gemini-3.1-flash-image",
                                        contents=prompt, #"A cyberpunk city at sunset",

                                        config=types.GenerateContentConfig(
                                            response_modalities=["IMAGE"]
                                        ),
                                    )
                                    if response:
                                        st.success("✓ Image Generated")
                                        # Display the generated image
                                        image_data = response.images[0]._image_bytes
                                        st.image(image_data, caption="Generated Image", use_container_width=True)

                                        # for part in response.candidates[0].content.parts:

                                        #     if part.inline_data:

                                        #         image = Image.open(
                                        #             BytesIO(part.inline_data.data)
                                        #         )

                                        #         image.save("city.png")
                                    else:
                                        st.warning("No images were generated. Please try a different prompt.")
                            except AttributeError:
                                # If imagen is not available, show user message
                                st.info("Gemini Text-to-Image generation is under development. Please select OpenAI for image generation.")
                            except Exception as e:
                                st.warning(f"Gemini image generation not available: {str(e)}. Please select OpenAI.")
                        elif model_choice == "Groq":
                            st.info("Groq does not support image generation in this Version. Please select OpenAI or Gemini.")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
            else:
                st.warning("Please describe the image you want to generate")

# ============ IMAGE TO TEXT ============
# All good - OpenAI, Gemini, Groq
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
                        if model_choice == "Gemini":
                            model = models["gemini"]
                        elif model_choice == "OpenAI":
                            model = models["openai"]
                        elif model_choice == "Groq":
                            model = models["groq"]
                        # elif model_choice == "HuggingFace":
                        #     st.info("HuggingFace models do not support image analysis in this version.")
                        #     st.stop()
                        
                        if model_choice == "Gemini":
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
                        else:  # OpenAI and Groq need image_url as an object
                            # https://console.groq.com/docs/vision#how-to-pass-images-from-urls-as-input
                            # "meta-llama/llama-4-scout-17b-16e-instruct" models supports multimodality,
                            #  while I previously used "llama-3.3-70b-versatile" which is a text-only model. 
                            message = HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": prompts.get(analysis_type, "Describe this image.")
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        }
                                    }
                                ]
                            )
                        
                        response = model.invoke([message])
                        
                        st.success("✓ Analysis Complete")


                        if model_choice == "Gemini":  # Only for Gemini3 Models, output comes as list
                            if isinstance(response.content, list):
                                response_content = response.content[0]['text']
                                st.info(response_content)
                            else:
                                st.info(response.content)
                        else:
                            st.info(response.content)
        
                        # st.info(response.content)
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
                        if model_choice == "Gemini":
                            model = models["gemini"]
                        elif model_choice == "OpenAI":
                            model = models["openai"]
                        elif model_choice == "Groq":
                            model = models["groq"]
                        # elif model_choice == "HuggingFace":
                        #     st.info("HuggingFace models do not support image analysis in this version.")
                        #     st.stop()
                        
                        prompts = {
                            "General Description": "Describe this image in a few sentences.",
                            "Detailed Analysis": "Provide a detailed analysis of this image.",
                            "Extract Text (OCR)": "Extract all text visible in this image.",
                            "Scene Understanding": "Analyze what's happening in this scene."
                        }
                        
                        if model_choice == "Gemini":
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
                        else:  # OpenAI and Groq need image_url as an object
                            message = HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": prompts.get(analysis_type, "Describe this image.")
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url
                                        }
                                    }
                                ]
                            )
                        
                        response = model.invoke([message])
                        st.success("✓ Analysis Complete")

                        if model_choice == "Gemini":  # Only for Gemini3 Models, output comes as list
                            if isinstance(response.content, list):
                                response_content = response.content[0]['text']
                                st.info(response_content)
                            else:
                                st.info(response.content)
                        else:
                            st.info(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============ TEXT TO AUDIO ============
# All good - OpenAI, Gemini
# For future reference: https://console.groq.com/docs/text-to-speech
elif modality_type == "Text to Audio":
    st.header("🔊 Text to Audio - Speech Synthesis")
    
    text_input = st.text_area(
        "Enter text to convert to speech:",
        height=150,
        placeholder="Enter the text you want to hear...",
        key="tts_input"
    )
    
    if model_choice == "OpenAI":
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
            if text_input.strip() and models["openai_client"]:
                with st.spinner("Generating audio..."):
                    try:
                        response = models["openai_client"].audio.speech.create(
                            model="tts-1-hd", # tts-1, https://developers.openai.com/api/docs/guides/text-to-speech
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
    
    elif model_choice == "Gemini":
        col1, col2 = st.columns(2)
        with col1:
            voice_name = st.selectbox(
                # https://ai.google.dev/gemini-api/docs/speech-generation#voices
                "Select voice:",
                ["Kore","Puck"],  # Kore - Female, Puck- Male voice
                help="Voice available in Gemini TTS"
            )
        
        with col2:
            speed = st.slider(
                "Speech speed:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        if st.button("🎙️ Generate Audio", use_container_width=True):
            if text_input.strip() and models.get("gemini_genai_client"):
                with st.spinner("Generating audio..."):
                    try:
                        audio_response = models["gemini_genai_client"].models.generate_content(
                            model= "gemini-3.1-flash-tts-preview",  #"gemini-2.5-flash-preview-tts" , # "gemini-2.0-flash-001",
                            contents=text_input,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=voice_name
                                        )
                                    )
                                )
                            )
                        )
                        
                        # Extract audio data

                        if audio_response.candidates and len(audio_response.candidates) > 0:                            
                            pcm_audio_data = audio_response.candidates[0].content.parts[0].inline_data.data

                            wav_buffer = io.BytesIO()

                            with wave.open(wav_buffer, "wb") as wf:
                                wf.setnchannels(1)      # mono
                                wf.setsampwidth(2)      # 16-bit
                                wf.setframerate(24000)  # 24 kHz
                                wf.writeframes(pcm_audio_data)

                            wav_bytes = wav_buffer.getvalue()
                            
                            
                            st.success("✓ Audio Generated")
                            # st.audio(audio_data, format="audio/mp3")

                            st.audio(wav_bytes, format="audio/wav")
                            
                            # Download button
                            # st.download_button(
                            #     label="📥 Download Audio",
                            #     data=audio_data,
                            #     file_name="generated_speech.mp3",
                            #     mime="audio/mp3"
                            # )

                            st.download_button(
                                label="📥 Download Audio(WAV)",
                                data=wav_bytes,
                                file_name="gemini_speech.wav",
                                mime="audio/wav"
                            )
                        else:
                            st.error("No audio data received from Gemini. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating audio: {e}")
            else:
                st.warning("Please enter some text to convert to speech")

# ============ AUDIO TO TEXT ============
# All good - OpenAI, Gemini
# For future reference: https://console.groq.com/docs/speech-to-text
elif modality_type == "Audio to Text":
    st.header("🎙️ Audio to Text - Transcription")
    
    audio_source = st.radio(
        "Choose audio source:",
        ["📤 Upload Audio File"],
        horizontal=True
    )
    
    if audio_source == "📤 Upload Audio File":
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
        )
        
        if audio_file:
            st.audio(audio_file)
            
            if model_choice == "OpenAI":
                if st.button("📝 Transcribe Audio", use_container_width=True):
                    if models["openai_client"]:
                        with st.spinner("Transcribing audio..."):
                            try:
                                transcript = models["openai_client"].audio.transcriptions.create(
                                    #https://developers.openai.com/api/docs/guides/speech-to-text
                                    model="whisper-1",  # gpt-4o-mini-transcribe
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
            
            elif model_choice == "Gemini":
                if st.button("📝 Transcribe Audio", use_container_width=True):
                    if models.get("gemini_genai_client"):
                        with st.spinner("Transcribing audio..."):
                            try:
                                # Encode audio to base64
                                audio_bytes = audio_file.read()
                                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                                
                                from langchain_core.messages import HumanMessage
                                message = HumanMessage(
                                    content=[
                                        {
                                            "type": "text",
                                            "text": "Transcribe the contents of this audio into English text."
                                        },
                                        {
                                            "type": "media",
                                            "mime_type": "audio/mp3",
                                            "data": base64_audio
                                        }
                                    ]
                                )
                                
                                response = models["gemini"].invoke([message])
                                
                                st.success("✓ Transcription Complete")
                                
                                if model_choice == "Gemini":  # Only for Gemini3 Models, output comes as list
                                    if isinstance(response.content, list):
                                        response_content = response.content[0]['text']
                                        st.info(response_content)
                                    else:
                                        st.info(response.content)
                                else:
                                    st.info(response.content)


                                # st.info(response.content)
                                # Download button
                                st.download_button(
                                    label="📋 Copy Transcription",
                                    data=response.content,
                                    file_name="transcription.txt",
                                    mime="text/plain"
                                )
                            except Exception as e:
                                st.error(f"Error transcribing audio: {e}")

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
if models:
    st.sidebar.success("✓ All models loaded successfully")
else:
    st.sidebar.warning("⚠️ Some models failed to load. Check API keys.")
