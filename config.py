from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.1-flash-lite"        # перевірте актуальну назву в AI Studio
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768