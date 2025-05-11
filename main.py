from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from service import start_image_task, get_image_task_status, enhance_prompt

app = FastAPI()

@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    return {"message": "Server is up 🎉"}


# ────────── request model for prompt enhancer ──────────
class PromptEnhanceRequest(BaseModel):
    prompt: str
    style: str | None = None     
    max_words: int = 60           

# ────────── enhancer endpoint ──────────
@app.post("/enhance_prompt/", status_code=status.HTTP_200_OK)
async def enhance_prompt_api(req: PromptEnhanceRequest):

    try:
        return enhance_prompt(
            prompt=req.prompt,
            style=req.style
        )
    except HTTPException as e:
        raise e
    
# ────────── request/response models ──────────
class ImageGenerationRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    style: str | None = None

# ────────── kick‑off endpoint ──────────
@app.post("/generate_image_flux/", status_code=status.HTTP_202_ACCEPTED)
async def generate_image_flux(req: ImageGenerationRequest):
    """
    Create a Flux txt2img job and return the task_id.
    The client should poll /image_task/{task_id} for status.
    """
    try:
        return start_image_task(
            prompt=req.prompt,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            style=req.style,
        )
    except HTTPException as e:
        raise e

# ────────── status endpoint ──────────
@app.get("/image_task/{task_id}", status_code=status.HTTP_200_OK)
async def image_task_status(task_id: str):
    """
    Return current status for a task_id.
    If status == 'success' the JSON will also contain image_url.
    """
    try:
        return get_image_task_status(task_id)
    except HTTPException as e:
        raise e
