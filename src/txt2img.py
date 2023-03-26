from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a close up of a woman's face with stars in the background, by Yoshio Markino, tumblr, the little mermaid, kiss, animated movie still, sleeping beauty AND a woman standing in the water holding a stick, a photo, by Doug Wildey, flickr, magic realism, tuesday weld in a pink bikini, film still dnd movie, on an island, motu"
image = pipe(prompt).images[0]
    
image.save("a close up of a woman's face with stars in the background, by Yoshio Markino, tumblr, the little mermaid, kiss, animated movie still, sleeping beauty AND a woman standing in the water holding a stick, a photo, by Doug Wildey, flickr, magic realism, tuesday weld in a pink bikini, film still dnd movie, on an island, motu.png")
