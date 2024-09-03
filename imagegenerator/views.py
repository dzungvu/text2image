from django.shortcuts import render
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(request):
    image_url = None
    if request.method == 'POST':
        text = request.POST.get('text')
        # Generate image from text
        image = pipe(text).images[0]
        image_path = f"media/generated_image.png"
        image.save(image_path)
        image_url = f"/{image_path}"
    return render(request, 'image_generator.html', {'image_url': image_url})