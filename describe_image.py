from google import genai
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def describe_image(file_path):
    try:
        img = Image.open(file_path)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[img, "Tell me about the object in image in detail"]
        )
        return response.text
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"
