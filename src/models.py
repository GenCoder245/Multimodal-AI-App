import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from openai import OpenAI
from google import genai

load_dotenv()

def load_models():
    models = {}
    try:
        models["gemini"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception:
        models["gemini"] = None

    try:
        models["openai"] = ChatOpenAI(model="gpt-5.4-nano")
        models["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        models["openai"] = None
        models["openai_client"] = None

    try:
        models["groq"] = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    except Exception:
        models["groq"] = None

    try:
        models["ollama"] = ChatOllama(model="gemma3:1b", temperature=0.7)
    except Exception:
        models["ollama"] = None

    try:
        models["gemini_genai_client"] = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception:
        models["gemini_genai_client"] = None

    return models