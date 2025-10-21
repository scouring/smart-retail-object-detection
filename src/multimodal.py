from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_scene_caption(image_path, scene_texts):
    image = Image.open(image_path)
    inputs = processor(text=scene_texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    predicted_index = logits_per_image.argmax().item()
    return scene_texts[predicted_index]
