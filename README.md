# Multimodal AI Application using Langchain

This Project contains a streamlit app `multimodal_app_new.py` which is used for processing multi-modal inputs and generate multi-modal outputs using LLM's in python.

There is also a modular Streamlit app that separates model loading, handlers and utilities etc mirroring the working of original `multimodal_app_new.py` but split into modules for production-readiness inside `src` folder.
 
    
    Built with:
    - 🦜 LangChain
    - 🤖 OpenAI, Gemini, Groq, Ollama, HuggingFace
    - 📊 Streamlit
    
    Features:
    - AI Chatbot (Text to Text)
    - Image Generation & Analysis
    - Speech Synthesis & Transcription
    - Multiple Model Providers
    - Local Models Support (Ollama)
## Setup 

Once you are inside the project's root directory follow the below steps for setting up the virutal environment, requirements and run the project files.

### Using `uv`

1. Initialize uv workspace: (Other python versions can also be used)
   - `uv init --python 3.11`
2. Create the virtual environment:
   - `uv venv`
3. Activate the virtual environment:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows cmd: `.venv\Scripts\activate.bat`
4. Install dependencies:
   - `uv add -r requirements.txt`
   or  `uv sync`


### Using standard `venv`

1. Create the virtual environment:
   - `python -m venv .venv`
2. Activate the virtual environment:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows cmd: `.venv\Scripts\activate.bat`
3. Install dependencies:
   - `pip install -r requirements.txt`

### Environment variables:

Make sure to set these environment variables inside a 
.env file

- `OPENAI_API_KEY` — OpenAI API key
- `GOOGLE_API_KEY` — Google Gemini / GenAI key
- `GROQ_API_KEY` — Groq API key (optional)
- `HF_API_TOKEN` — HuggingFace token (optional)


Note:
  - `streamlit run app.py` - For running the modularized source code(code inside .\src\)

  - `streamlit run multimodal_app_new.py` - For running the entire source code present inside a single file.



## Repository explanation

1. **multimodal_app_new.py** -  Current version of the multi-modal-ai project. Entire source code present inside a single python file.

2. **multimodal_app_old.py** -  Earlier version of the multi-modal-ai project. Just for reference alone.

3. **multi_modality_notebook.ipynb** - The .ipynb notebook file used for experimentation before incorporating in the python files.

4. **src folder** - The modularized version of the **multimodal_app_new.py**

5. **gui_demo1.png**, **gui_demo2.png** - Screenshots of the GUI of the developed streamlit app.

6. **files_generated folder** - Some sample image, audio files generated using the developed code.