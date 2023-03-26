import requests
import base64
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# Encode the image data in base64 format
image_data = base64.b64encode(image.tobytes()).decode("utf-8")

# Upload the image to imgbb
api_key = "your_api_key"
response = requests.post("https://api.imgbb.com/1/upload", 
                         data={"key": api_key, "image": image_data})

# Print the response
if response.status_code == 200:
    image_url = response.json()["data"]["url"]
    print("Image URL: ", image_url)
else:
    print("Error: ", response.text)
