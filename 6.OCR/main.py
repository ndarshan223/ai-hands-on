from fastapi import FastAPI, HTTPException
import cv2 as cv
from PIL import Image
import requests
import numpy as np
import utils
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
api_key = os.getenv("API_KEY")

@app.get("/")
def home():
    return {"message": "Hello, this is my API!"}

@app.get("/add")
def add(link: str):
    try:
        if not api_key:
            raise HTTPException(status_code=500, detail="API_KEY not found in environment variables")
        
        http_response = requests.get(link)
        img_array = np.frombuffer(http_response.content, np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)

        extracted_text = utils.extract_text_from_image(img)
        gemini_response = utils.structure_with_gemini(extracted_text, api_key)
        output = utils.format_response(gemini_response)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # uvicorn.run(app, host="0.0.0.0", port=port)