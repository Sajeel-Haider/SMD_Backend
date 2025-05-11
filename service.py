import os, requests, json
from fastapi import HTTPException
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

PI_API_KEY      = os.getenv("PI_API_KEY")
GOOGLE_API_KEY  = os.getenv("GEMINI_API")
MODEL_NAME      = "Qubico/flux1-schnell"

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")

# ─── Add just below the existing imports / Gemini setup ───────────────────────
def enhance_prompt(
    prompt: str,
    style: str | None = None,
    max_words: int = 30,
) -> dict:
    base_instruction = (
        "Rewrite the following image prompt so that it is vivid, "
        "specific, and richly descriptive. Keep it under "
        f"{max_words} words."
    )
    if style:
        base_instruction += f" Make sure the rewrite clearly evokes a {style} style."

    instruction = (
        f"{base_instruction}\n\n"
        f"Original prompt: \"{prompt}\"\n\n"
        f"Enhanced:"
    )

    try:
        resp = gemini.generate_content(instruction)
        enhanced = resp.text.strip().strip('"\'')

        # Gemini can sometimes return an empty string on rare failures
        if not enhanced:
            enhanced = f"{prompt}, {style} style" if style else prompt

        return {"enhanced_prompt": enhanced}
    except Exception as e:
        # Surface LLM error in HTTPException so FastAPI returns 500
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

# ─────────────────────────────────── helpers ───────────────────────────────────
def _enhance_prompt(prompt: str, style: str | None) -> str:
    """Return the prompt rewritten in *style* (falls back to simple concat)."""
    if not style:
        return prompt

    instruction = (
        f"Rewrite the following image prompt so it clearly evokes a {style} style. "
        f"Keep it under 60 words.\nPrompt: \"{prompt}\"\nEnhanced:"
    )
    try:
        resp = gemini.generate_content(instruction)
        enhanced = resp.text.strip().strip('"\'')

        return enhanced or f"{prompt}, {style} style"
    except Exception:
        return f"{prompt}, {style} style"


headers = {
    "X-API-Key": PI_API_KEY,
    "Content-Type": "application/json",
}

# ───────────────────────────── public service API ─────────────────────────────
def start_image_task(
    prompt: str,
    guidance_scale: float,
    width: int,
    height: int,
    style: str | None,
) -> dict:
    """Submits a txt2img job to Pi API and returns the task_id."""
    body = {
        "model": MODEL_NAME,
        "task_type": "txt2img",
        "input": {
            "prompt": _enhance_prompt(prompt, style),
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
        },
    }

    resp = requests.post(
        "https://api.piapi.ai/api/v1/task", headers=headers, json=body, timeout=60
    )
    if not resp.ok:
        raise HTTPException(resp.status_code, resp.text)

    task_id = resp.json()["data"]["task_id"]
    return {"task_id": task_id}

def get_image_task_status(task_id: str) -> dict:
    """Fetch Pi API task status.  
    • If status is *success* or *completed* → return the full Pi API response JSON  
    • Otherwise → return a lightweight status object for polling
    """
    resp = requests.get(
        f"https://api.piapi.ai/api/v1/task/{task_id}",
        headers=headers,
        timeout=60,
    )

    if not resp.ok:
        raise HTTPException(resp.status_code, resp.text)

    api_json = resp.json()          # full response (contains code, data, message)
    data = api_json.get("data", {})
    status = data.get("status", "").lower()

    # Pi API sometimes uses "completed" instead of "success"
    if status in {"success", "completed"}:
        # give caller the *entire* Pi API payload, exactly as received
        return api_json

    # anything else → still polling or failed
    result = {
        "status": status,           # queued, running, failed, etc.
        "task_id": task_id,
    }

    # surface server‑side error message if present
    if "error" in data and data["error"].get("message"):
        result["error"] = data["error"]["message"]

    return result

