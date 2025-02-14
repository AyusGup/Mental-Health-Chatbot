import os
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel
from model import Chatbot

# Load environment variables
load_dotenv()

PORT = int(os.getenv("PORT", "5000"))
SESSION_EXPIRY = int(os.getenv("SESSION_EXPIRY", "3600"))

app = FastAPI()

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
chatbot = Chatbot()

class TextRequest(BaseModel):
    user_id: str
    text: str

@app.post("/analyze-text/")
async def analyze_text(request: TextRequest):
    """
    API endpoint to process text using the chatbot model.
    """
    try:
        text = request.text
        user_id = request.user_id
        print(text, user_id)

        response = chatbot.generate_response(user_id, text)

        # Log response
        timestamp = datetime.now().isoformat()
        record = {
            "text": text,
            "response": response,
            "created_at": timestamp,
        }
        print(f"Chatbot Record: {record}")

        return JSONResponse(status_code=200, content={"response": response})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Mental Health Chatbot API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
