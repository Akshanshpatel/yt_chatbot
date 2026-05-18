from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from google import genai
from langchain_core.embeddings import Embeddings
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# VIDEO ID

video_id = "xwhJfqIyoBY"

# FETCH TRANSCRIPT

try:
    transcript_list = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en", "hi"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

except TranscriptsDisabled:
    print("No transcript available")
    exit()

# TEXT SPLITTING

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Total Chunks:", len(chunks))

# GEMINI CLIENT

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# CUSTOM EMBEDDING CLASS

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config={
                    "output_dimensionality": 768
                }
            )

            embeddings.append(
                response.embeddings[0].values
            )

        return embeddings

    def embed_query(self, text):

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                "output_dimensionality": 768
            }
        )

        return response.embeddings[0].values

# EMBEDDING MODEL

embedding_model = GeminiEmbeddings()

# UNIQUE FAISS PATH

FAISS_PATH = f"FAISS/{video_id}"

# Create FAISS folder if not exists
os.makedirs("FAISS", exist_ok=True)

# LOAD OR CREATE VECTOR DB

if os.path.exists(FAISS_PATH):

    print(f"Loading existing FAISS for video: {video_id}")

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

else:

    print(f"Creating new FAISS for video: {video_id}")

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_store.save_local(FAISS_PATH)

    print("FAISS saved successfully.")

# VECTOR STORE INFO

print("Vector store ready.")
print("Total vectors:", vector_store.index.ntotal)

# RETRIEVER--> similarity or MMR (Maximum Marginal Relevance)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

# LLM
model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.2)

# Prompt

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
    - Then provide a general knowledge answer
      using your own understanding.

    Keep answers short and clear.

    Transcript Context:
    {context}

    User Question:
    {question}
    """,
    input_variables=['context', 'question']
)

## Question of user

question="Tell 3 jokes out of this video"
relevant_docs=retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

### Below lines are better represented in chains

# final_prompt = prompt.invoke({"context": context_text, "question": question})
# answer = model.invoke(final_prompt)
# print(answer.content)

parser=StrOutputParser()

chain = prompt | model | parser

answer= chain.invoke({"context": context_text, "question": question})
print(answer)