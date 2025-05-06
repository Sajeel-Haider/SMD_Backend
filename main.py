from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .service import generate_image_openai, generate_image_flux

app = FastAPI()

class ImageGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate_image_openai/")
async def generate_image_with_openai(request: ImageGenerationRequest):
    try:
        image_url = generate_image_openai(request.prompt)
        return {"image_url": image_url}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

@app.post("/generate_image_flux/")
async def generate_image_with_flux(request: ImageGenerationRequest):
    try:
        image_data = generate_image_flux(request.prompt)
        # Save the image data to a file (you can modify this to use cloud storage if needed)
        with open("generated_image.png", "wb") as file:
            file.write(image_data)
        return {"image_url": "generated_image.png"}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
