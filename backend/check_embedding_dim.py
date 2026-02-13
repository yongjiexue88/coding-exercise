
from services.embedding import EmbeddingService
from config import settings
import google.generativeai as genai

def check():
    import google.generativeai as genai
    model = "models/gemini-embedding-001"
    if settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
    
    try:
        vec = genai.embed_content(
            model=model,
            content="Hello world",
            task_type="retrieval_document",
            output_dimensionality=768
        )["embedding"]
        print(f"Model: {model}")
        print(f"Truncated dimension: {len(vec)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
