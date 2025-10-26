import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    # Configure the Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    
    genai.configure(api_key=api_key)

    print("--- Available Models for Your API Key ---")
    
    # List all models and their supported methods
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model Name: {model.name}")

    print("-----------------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")