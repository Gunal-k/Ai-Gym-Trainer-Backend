# chat_service.py
import os, uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")
genai.configure(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chatbot")
async def chat_with_bot(request: ChatRequest):
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        response = model.generate_content(request.message)
        if response.text:
            return {"reply": response.text}
        else:
            raise HTTPException(status_code=500, detail="Failed to get a valid response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) # Note the different port