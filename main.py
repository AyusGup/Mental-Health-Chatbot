import os
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel

# Load environment variables
load_dotenv()

PORT = int(os.getenv("PORT", "5000"))

app = FastAPI()

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load fine-tuned model and tokenizer
MODEL_NAME = "thrishala/mental_health_chatbot"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

def extract_chatbot_response(text):
    """Extract chatbot's response after [/INST]"""
    parts = text.split("[/INST]")  
    if len(parts) > 1:
        response = parts[1].strip()
        sentences = response.split(". ")  # Extract only the first complete sentence
        return sentences[0] + "." if sentences else response
    return "No response found."

def generate_response(prompt, max_new_tokens=100): 
    """Generates response using the chatbot model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,  
                max_new_tokens=max_new_tokens,  # Limit response length
                repetition_penalty=1.2,  
                temperature=0.7,         
                top_p=0.8,  
                top_k=30
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return extract_chatbot_response(response)  # Clean response

    except Exception as e:
        return f"Error generating response: {str(e)}"

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text/")
async def analyze_text(request: TextRequest):
    """
    API endpoint to process text using the chatbot model.
    """
    try:
        text = request.text
        response = generate_response(text)

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
