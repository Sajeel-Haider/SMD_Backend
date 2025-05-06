import os
import openai
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("HF_API_TOKEN")

# Hugging Face FLUX MODEL ID
MODEL_ID = "black-forest-labs/FLUX.1-dev"

def get_image_generation_prompt(response: dict) -> str:
    return response.get('image_generation_prompt', '')

def generate_image_openai(prompt: str):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }

    data = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post(
        'https://api.openai.com/v1/images/generations',
        json=data,
        headers=headers
    )

    if response.status_code == 200:
        image_url = response.json()['data'][0]['url']
        return image_url
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())

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
        f"https://api-inference.huggingface.co/models/{MODEL_ID}", 
        headers=headers, 
        json=data
    )

    if response.status_code == 200:
        image_data = response.content
        return image_data
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)
