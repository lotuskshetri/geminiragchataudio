# geminiragchataudio
Implementation of RAG application using Gemini flash app 2 api,with added functionalities of audio input and output using whisper and gTTS respectively.

clone the repo and save a .env file with your GEMINI API KEY as 

.env
GOOGLE_API_KEY="YOUR_API_KEY"
GEMINI_ENDPOINT=https://api.gemini.flash


Run the application with streamlit run app.py

Upload a file and use submit and process button to create embeddings.

You can either select audio or text form for prompt. The output will be produced in both text and audio.
