from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a man standing next to a bike next to a body of water, 2 0 2 1 cinematic 4 k framegrab, at purple sunset, a man wearing a backpack, midsommar AND a man riding a wave on top of a surfboard, inspired by William Trost Richards, purism, an extreme long shot wide shot, # oc, 1 0 / 1 0, oregon"
image = pipe(prompt).images[0]
    
image.save("photoooss.png")
