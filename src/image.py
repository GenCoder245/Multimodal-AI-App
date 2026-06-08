import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from google.genai import types

def render_text_to_image(models, model_choice):
    st.header("🖼️ Text to Image - Generation")
    prompt = st.text_area("Describe the image you want to generate:", height=150, placeholder="Example: A serene landscape with mountains and a lake at sunset...", key="image_prompt")
    quality = st.selectbox("Image Quality:", ["low", "medium", "high"],
        help="Lower quality generates faster")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎨 Generate Image", use_container_width=True):
            if not prompt.strip():
                st.warning("Please describe the image you want to generate")
                return
            with st.spinner("Generating image..."):
                try:
                    if model_choice == "OpenAI":
                        response = models["openai"].invoke(
                            [HumanMessage(content=prompt)],
                            tools=[{"type": "image_generation", "quality": quality}],
                            tool_choice={"type": "image_generation"}
                        )

                        # Extract and display the image
                        for block in getattr(response, "content_blocks", []):
                            if block.get("type") == "image":
                                image_base64 = block["base64"]
                                st.image(base64.b64decode(image_base64), caption="Generated Image", use_container_width=True)
                                st.success("✓ Image Generated")
                    elif model_choice == "Gemini":
                        client = models.get("gemini_genai_client")
                        if not client:
                            st.warning("Gemini client not available.")
                            return
                        response = client.models.generate_content(
                            model="gemini-3.1-flash-image",
                            contents=prompt,
                            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
                        )
                        if response and getattr(response, "images", None):
                            st.image(response.images[0]._image_bytes, caption="Generated Image", use_container_width=True)
                            st.success("✓ Image Generated")
                        else:
                            st.warning("No images returned.")
                    else:
                        st.info("Selected provider does not support image gen in this setup.")
                except Exception as e:
                    st.error(f"Error generating image: {e}")

def render_image_to_text(models, model_choice):
    st.header("🔍 Image to Text - Analysis")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png","gif","webp"])
    if not uploaded_file:
        image_url = st.text_input("Or paste image URL:")
        if image_url:
            st.image(image_url, use_container_width=True)
            show_analysis_controls(models, model_choice, image_url=image_url)
    else:
        st.image(uploaded_file, use_container_width=True)
        image_bytes = uploaded_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        show_analysis_controls(models, model_choice, base64_data=f"data:image/png;base64,{base64_image}")

def show_analysis_controls(models, model_choice, image_url=None, base64_data=None):
    analysis_type = st.selectbox("Type of Analysis:", ["General Description", "Detailed Analysis", "Extract Text (OCR)", "Scene Understanding"])
    prompt_map = {
        "General Description": "Describe this image in a few sentences.",
        "Detailed Analysis": "Provide a detailed analysis of this image including elements, colors, composition, and mood.",
        "Extract Text (OCR)": "Extract and transcribe all text visible in this image.",
        "Scene Understanding": "Analyze the scene; what's happening and main elements?"
    }
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                if model_choice == "Gemini":
                    model = models.get("gemini")
                    content = [
                        {"type": "text", "text": prompt_map[analysis_type]},
                        {"type": "image_url", "image_url": image_url or base64_data}
                    ]
                else:
                    model = models.get("openai") or models.get("groq")
                    content = [
                        {"type": "text", "text": prompt_map[analysis_type]},
                        {"type": "image_url", "image_url": {"url": image_url or base64_data}}
                    ]
                if not model:
                    st.error("Selected model isn't available.")
                    return
                response = model.invoke([HumanMessage(content=content)])
                if model_choice == "Gemini" and isinstance(response.content, list):
                    st.info(response.content[0].get("text", ""))
                else:
                    st.info(response.content)
                st.success("✓ Analysis Complete")
            except Exception as e:
                st.error(f"Error analyzing image: {e}")