

from latentdiffusion.ldm.models.diffusion.ddim import DDIMSampler
from latentdiffusion.ldm.extras import load_model_from_config, load_training_dir

import torch
import clip.clip as clip
from PIL import Image
# rest of the code here
import requests
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP clip_model
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
ckpt="v1-5-pruned-emaonly.ckpt"
config="v1-inference.yaml"
model = load_model_from_config(config, ckpt, device=device, verbose=False)

ddim_sampler = DDIMSampler(model) 

def get_clip_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).to(device)
    tensor = torch.unsqueeze(tensor, 0)

    with torch.no_grad():
        clip_vector = clip_model.encode_image(tensor).float()
    return clip_vector

def generate_image(clip_vector, scale=1.0, steps=30):
    with torch.no_grad():
        shape = [4, 512 // 8, 512 // 8]
        start_code = torch.randn(1, *shape, device=device)
        samples, _ = ddim_sampler.sample(S=steps,
                                         conditioning=clip_vector.unsqueeze(0),
                                         batch_size=1,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         x_T=start_code)
        generated_image = model.decode_first_stage(samples)[0]
    return generated_image

# save the image to a file
clip_vector=get_clip_vector("/home/ubuntu/images/oussamaammar/00001.jpg")
generated_image=generate_image(clip_vector, scale=1.0, steps=30)
generated_image = generated_image.to('cpu').numpy()
generated_image = generated_image.transpose((1, 2, 0)) * 255.0
generated_image = generated_image.astype('uint8')
image_pil = Image.fromarray(generated_image)
image_pil.save("generated_image.jpg")

# upload the image to ImgBB and get the URL
with open("generated_image.jpg", "rb") as file:
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": "your_api_key",
        "image": base64.b64encode(file.read()),
    }
    res = requests.post(url, payload)
    print(res.json())


