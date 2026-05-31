from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from google import genai
from dotenv import load_dotenv

import requests
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
            contents=texts,
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

    os.makedirs("FAISS", exist_ok=True)

    faiss_path = f"FAISS/{video_id}"

    # -------------------------
    # Load Existing FAISS
    # -------------------------

    if os.path.exists(faiss_path):

        print("Loading existing FAISS...")

        vector_store = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:

        print("FAISS not found. Fetching transcript...")

        try:

            response = requests.get(
                f"https://api.supadata.ai/v1/youtube/transcript?videoId={video_id}",
                headers={
                    "x-api-key": os.getenv("SUPADATA_API_KEY")
                },
                timeout=30
            )

            if response.status_code != 200:

                raise Exception(
                    f"Supadata Error: {response.text}"
                )

            data = response.json()

            transcript = " ".join(
                chunk["text"]
                for chunk in data["content"]
            )

            if not transcript.strip():

                raise Exception(
                    "Transcript not available"
                )

        except Exception as e:

            raise Exception(
                f"Failed to fetch transcript: {str(e)}"
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents(
            [transcript]
        )

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

