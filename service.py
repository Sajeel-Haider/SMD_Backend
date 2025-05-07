import os
import tempfile
 
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

import google.generativeai as genai
import os
import json
import shutil

 
load_dotenv()


 
API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GEMINI_API")
genai.configure(api_key=GOOGLE_API_KEY)
# Hugging Face FLUX MODEL ID
HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def split_prompt(prompt: str):
    instruction = (
    f"The user wants to create a short video from a sequence of 4 images. Based on their overall video idea below, generate 4 distinct sub-prompts. "
    f"Each sub-prompt should describe a scene for one image in the sequence. The scenes should flow logically to create a mini-narrative or show clear progression. "
    f"Consider how elements like character, setting, and mood might carry through or evolve across the sequence to ensure visual coherence in the final video.\n\n"
    f"User's Video Idea: \"{prompt}\"\n\n"
    f"Return the result as JSON in the following format:\n"
    f'{{"subprompts": ["...", "...", "...", "..."]}}'
)

    try:
        response = model.generate_content(instruction)
        text = response.text
 
        start = text.find('{')
        end = text.rfind('}') + 1
        json_str = text[start:end]

        return json.loads(json_str)

    except Exception as e:
        return {"error": str(e)}

 

def generate_image_flux(prompt: str):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt,
        "parameters": {
            "height": 1024,
            "width": 1024,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
        headers=headers,
        json=data
    )

    if response.ok:
        image_path = "generated_image_flux.png"
        with open(image_path, "wb") as f:
            f.write(response.content)

        return {"image_url": image_path}

    raise HTTPException(status_code=response.status_code, detail=response.text)
def generate_video_flux(prompt: str):
    print("Calling split_prompt...")
    split_res = split_prompt(prompt)
    subprompts = split_res.get("subprompts", [])
    print(f"Successfully split into {len(subprompts)} sub-prompts.")

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="video_images_")
        print(f"Created temporary directory: {temp_dir}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create temporary directory: {e}")

    generated_image_paths = []
    temp_flux_output_name = "generated_image_flux.png"
    print("Starting image generation for sub-prompts...")

    for i, subprompt in enumerate(subprompts):
        scene_number = i + 1
        print(f"Generating image {scene_number}/4 for prompt: '{subprompt}'")
        try:
            image_gen_result = generate_image_flux(subprompt)

            temp_source_path = temp_flux_output_name
            if not os.path.exists(temp_source_path):
                print(f"Warning: generate_image_flux did not create the expected temporary file '{temp_source_path}' for prompt: '{subprompt}'")
                continue

            final_image_name = f"scene_{scene_number}.png"
            final_image_path = os.path.join(temp_dir, final_image_name)
            shutil.move(temp_source_path, final_image_path)
            print(f"Successfully saved image {scene_number} to '{final_image_path}'")
            generated_image_paths.append(final_image_path)
        except HTTPException as e:
            print(f"Error generating image {scene_number} for prompt '{subprompt}': HTTP Error {e.status_code} - {e.detail}")
        except Exception as e:
            print(f"An unexpected error occurred generating image {scene_number}: {e}")
            if os.path.exists(temp_source_path):
                try:
                    os.remove(temp_source_path)
                    print(f"Cleaned up temporary file '{temp_source_path}'.")
                except Exception as cleanup_e:
                    print(f"Error during cleanup of '{temp_source_path}': {cleanup_e}")

    return {"image_paths": generated_image_paths}
    
 