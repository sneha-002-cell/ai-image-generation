from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe.to("cuda")

prompt = "A futuristic cyberpunk city, digital art"

image = pipe(prompt).images[0]
image.save("generated_image.png")

print("Image generated successfully!")
