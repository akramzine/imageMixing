import requests
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# Save the image to a local file
image_path = "astronaut_rides_horse.png"
image.save(image_path)

# Upload the image to imgbb
api_key = "a06c97753bd86796d1159e4ca7f1efc2"
with open(image_path, "rb") as f:
    response = requests.post("https://api.imgbb.com/1/upload", 
                             data={"key": api_key, "image": f})
    image_url = response.json()["data"]["url"]

# Print the image URL
print("Image URL: ", image_url)
