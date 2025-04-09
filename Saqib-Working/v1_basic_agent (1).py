import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(os.getenv('MODEL'))
agent = {"name": "Poem Writer","instructions": "You are an expert Poem Writer."}
response = model.generate_content(f"{agent['instructions']}\n\n{"Write a 4 sentence poem about the Sadness"}")
print(response.text)


