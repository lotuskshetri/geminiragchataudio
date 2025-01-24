import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from gtts import gTTS
import pyaudio
import wave
import whisper
import torch
from dotenv import load_dotenv  
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load environment variables
load_dotenv()

# Configure API key for Gemini
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Whisper model for speech-to-text
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base.en", device=device)

# Function: Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function: Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function: Generate embeddings using Gemini
def generate_embeddings(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function: Get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: {question}\n
    Answer:
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def answer_question(context, question):
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt_template.format(context=context, question=question)
        )
        return response.text

    return answer_question

# Function: Speech-to-text
def speech_to_text():
    # Audio configuration
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    st.info("Listening for your question (Press Ctrl+C to stop)...")

    try:
        # Capture 5 seconds of audio
        frames = []
        for _ in range(0, int(RATE / CHUNK * 5)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Save the audio data to a temporary file
        wav_output = "temp_audio.wav"
        wf = wave.open(wav_output, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Transcribe the audio
        result = whisper_model.transcribe(wav_output, language="en")
        return result["text"]

    except KeyboardInterrupt:
        st.error("Stopped listening.")
        return None

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Function: Text-to-speech
def text_to_speech(text):
    language = "en"
    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file

# Streamlit interface
def main():
    st.set_page_config("Chat PDF with Speech-to-Text and Text-to-Speech")
    st.title("Chat with PDF using Speech & Text 💬")
    
    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Upload and Process PDFs:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                generate_embeddings(text_chunks)
                st.success("PDF processing and indexing done!")

    # Input section
    st.subheader("Ask your question:")
    mode = st.radio("Input Mode:", ("Type your question", "Speak your question"))

    # User Question
    user_question = ""
    if mode == "Type your question":
        user_question = st.text_input("Enter your question:")
    elif mode == "Speak your question":
        if st.button("Record Question"):
            with st.spinner("Processing your speech..."):
                user_question = speech_to_text()
                st.success(f"Transcribed Question: {user_question}")

    # Answering the question
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        context = " ".join([doc.page_content for doc in docs])
        answer_question = get_conversational_chain()
        response = answer_question(context=context, question=user_question)

        st.write("Answer:", response)

        # Text-to-speech for the response
        audio_file = text_to_speech(response)
        st.audio(audio_file, format="audio/mp3", start_time=0)

if __name__ == "__main__":
    main()
