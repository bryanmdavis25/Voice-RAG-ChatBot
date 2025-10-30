import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import base64

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎙️ RAG Voice Chatbot", page_icon="🎧", layout="centered")

st.markdown(
    """
    <style>
    body { background-color: #F8F9FB; }
    .title { text-align: center; color: #4B0082; font-size: 2.2em; font-weight: bold; }
    .subtitle { text-align: center; color: gray; margin-bottom: 20px; }
    .chat-box { background: #fff; padding: 20px; border-radius: 10px; margin-top: 10px; box-shadow: 0px 3px 8px rgba(0,0,0,0.08); }
    .user-msg { background-color: #E8E8FF; padding: 12px; border-radius: 10px; margin-bottom: 5px; color: #333; }
    .bot-msg { background-color: #FFF6E8; padding: 12px; border-radius: 10px; margin-bottom: 5px; color: #333; border-left: 5px solid #4B0082; }
    audio { width: 100%; outline: none; margin-top: 8px; }
    </style>

    <div class='title'>🎙️ RAG Voice Chatbot</div>
    <p class='subtitle'>Speak naturally – get intelligent, spoken answers from your PDFs.</p>
    <hr style='border:1px solid #ddd'>
    """,
    unsafe_allow_html=True
)

# --- Initialize Vector DB once ---
@st.cache_resource
def init_vector_db():
    file_path = r"C:\\Users\\New Stech Computer\\Desktop\\Voice Chatbot\\Here.pdf"
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_text = " ".join([p.page_content for p in pages])
    chunks = text_splitter.split_text(all_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "voice-rag-index"
    dimension = 3072

    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        vectors.append({"id": f"chunk-{i}", "values": vector, "metadata": {"text": chunk}})
    index.upsert(vectors)
    return index, embeddings

index, embeddings = init_vector_db()

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Voice Input Section ---
st.markdown("### 🎧 Ask Your Question")

audio_obj = mic_recorder(
    start_prompt="🎙️ Click to Speak",
    stop_prompt="⏹️ Stop Recording",
    key="recorder",
    just_once=False
)

# --- Process New Question ---
if audio_obj is not None:
    if isinstance(audio_obj, dict) and "bytes" in audio_obj and audio_obj["bytes"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_obj["bytes"])
            temp_audio_path = temp_audio.name

        # --- Transcribe user voice ---
        with st.spinner("🕑 Transcribing your question..."):
            with open(temp_audio_path, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3-turbo",
                    language="en",
                )
        question = transcription.text.strip()

        # --- Retrieve Context from Pinecone ---
        q_embed = embeddings.embed_query(question)
        query_results = index.query(vector=q_embed, top_k=3, include_metadata=True)
        context = "\n".join([m["metadata"]["text"] for m in query_results["matches"]])

        # --- Generate Assistant Answer ---
        with st.spinner("🤖 Thinking..."):
            prompt = f"""
            Use the following context to answer the question concisely.

            Context:
            {context}

            Question:
            {question}
            """
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        answer_text = response.choices[0].message.content.strip()

        # --- Generate TTS ---
        with st.spinner("🎤 Speaking the answer..."):
            tts_response = groq_client.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=answer_text,
                response_format="wav"
            )
            audio_bytes = tts_response.read()
            b64_audio = base64.b64encode(audio_bytes).decode()

        # --- Store in Chat History ---
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer_text,
            "audio_b64": b64_audio
        })

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    st.markdown(f"""
        <div class="chat-box">
            <div class="user-msg"><b>🧑 You:</b> {chat['question']}</div>
            <div class="bot-msg"><b>🤖 Assistant:</b> {chat['answer']}
                <audio autoplay controls>
                    <source src="data:audio/wav;base64,{chat['audio_b64']}" type="audio/wav">
                </audio>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    "<hr><p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit, OpenAI, Groq & Pinecone</p>",
    unsafe_allow_html=True
)
