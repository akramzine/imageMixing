

from latentdiffusion.ldm.models.diffusion.ddim import DDIMSampler
from latentdiffusion.ldm.extras import load_model_from_config, load_training_dir

import torch
import clip.clip as clip
from PIL import Image
# rest of the code here


device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP clip_model
clip_model, preprocess = clip.load('ViT-B/32', device=device)
clip_model.eval()
ckpt="v1-5-pruned-emaonly.ckpt"
config="v1-inference.yaml"
model = load_model_from_config(config, ckpt, device=device, verbose=False)

ddim_sampler = DDIMSampler(model) 

def get_clip_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    # resize image
    image = image.resize((768, 320), resample=Image.LANCZOS) 
    tensor = preprocess(image).to(device)
    tensor = torch.unsqueeze(tensor, 0)

    with torch.no_grad():
        clip_vector = clip_model.encode_image(tensor).float()
    return clip_vector

def generate_image(clip_vector, scale=1.0, steps=30):
    with torch.no_grad():
        shape = [4, 768 // 8, 320 // 8]
        print(clip_vector.shape)
        start_code = torch.randn(1, *shape, device=device)
        samples, _ = ddim_sampler.sample(S=steps,
                                         conditioning=clip_vector.unsqueeze(0),
                                         batch_size=1,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         x_T=start_code)
        generated_image = ddim_sampler.decode_first_stage(samples)[0]
    return generated_image
print(generate_image(get_clip_vector("/home/ubuntu/images/oussamaammar/00001.jpg"),scale=1.0, steps=30))


