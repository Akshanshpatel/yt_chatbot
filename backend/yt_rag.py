from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from google import genai
from dotenv import load_dotenv

import os

load_dotenv()

# -------------------------
# Gemini Client
# -------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -------------------------
# Embeddings
# -------------------------

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,  # list of chunks
            config={
                "output_dimensionality": 768
            }
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def embed_query(self, text):

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                "output_dimensionality": 768
            }
        )

        return response.embeddings[0].values


embedding_model = GeminiEmbeddings()

# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# -------------------------
# Prompt
# -------------------------

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant for answering
questions about a YouTube video transcript.

First check whether the answer exists in the
provided transcript context.

If the answer is found in the transcript:
- Answer using the transcript information.

If the answer is NOT found in the transcript:
- Clearly say that the answer was not found
in the video transcript.
- Then provide a general knowledge answer.

IMPORTANT:
- Always answer in English.
- If the transcript is in Hindi or any other language,
  translate the information into English before answering.

Keep answers concise.

Transcript Context:
{context}

User Question:
{question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

# -------------------------
# Load/Create Retriever
# -------------------------

def load_video(video_id):

    try:

        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "hi"]
        )

        transcript = " ".join(
            chunk.text
            for chunk in transcript_list
        )

    except TranscriptsDisabled:
        raise Exception("Transcript not available")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    os.makedirs("FAISS", exist_ok=True)

    faiss_path = f"FAISS/{video_id}"

    if os.path.exists(faiss_path):

        print("Loading existing FAISS...")

        vector_store = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:

        print("Creating new FAISS...")

        vector_store = FAISS.from_documents(
            chunks,
            embedding_model
        )

        vector_store.save_local(
            faiss_path
        )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
        "k": 8,
        "fetch_k": 20
        }   
    )

    return retriever

# -------------------------
# QA Function
# -------------------------

def ask_question(retriever, question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return answer