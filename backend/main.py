from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from utils import extract_video_id
from yt_rag import load_video, ask_question

app = FastAPI()

# -------------------------
# CORS
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# In-memory retriever
# -------------------------

current_retriever = None
current_video_id = None

# -------------------------
# Request Models
# -------------------------

class VideoRequest(BaseModel):
    url: str


class QuestionRequest(BaseModel):
    question: str

# -------------------------
# Home Route
# -------------------------

@app.get("/")
def home():
    return {
        "message": "YouTube Chatbot Backend Running"
    }

# -------------------------
# Load Video
# -------------------------

@app.post("/load-video")
def load_video_endpoint(data: VideoRequest):

    global current_retriever
    global current_video_id

    try:

        video_id = extract_video_id(
            data.url
        )

        current_retriever = load_video(
            video_id
        )

        current_video_id = video_id

        return {
            "success": True,
            "video_id": video_id,
            "message": "Video loaded successfully"
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

# -------------------------
# Ask Question
# -------------------------

@app.post("/ask")
def ask_endpoint(data: QuestionRequest):

    global current_retriever

    if current_retriever is None:

        raise HTTPException(
            status_code=400,
            detail="Load a video first"
        )

    try:

        answer = ask_question(
            current_retriever,
            data.question
        )

        return {
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )