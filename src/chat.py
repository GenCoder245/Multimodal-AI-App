import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os

def _select_model_instance(models, model_choice):
    if model_choice == "Gemini":
        return models.get("gemini")
    if model_choice == "OpenAI":
        return models.get("openai")
    if model_choice == "Groq":
        return models.get("groq")
    if model_choice == "Ollama":
        return models.get("ollama")
    if model_choice == "HuggingFace":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        hf_llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V4-Flash",
            task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN")
        )
        return ChatHuggingFace(llm=hf_llm)
    return None

def render_chat(models, model_choice):
    st.header("💬 Text to Text - AI Chatbot")
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
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.spinner("Generating response..."):
            try:
                model = _select_model_instance(models, model_choice)
                if model is None:
                    st.error("Selected model is not available.")
                    return

                response = model.invoke(st.session_state.chat_history)
                # Only for Gemini3 Models, output comes as list
                if model_choice == "Gemini" and isinstance(response.content, list):
                    content = response.content[0].get("text", "")
                else:
                    content = response.content

                # Add assistant response to history
                st.session_state.chat_history.append(AIMessage(content=content))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()