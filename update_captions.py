import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_captions():
    caption_file = "captions.json"
    if os.path.exists(caption_file):
        with open(caption_file, "r") as f:
            return json.load(f)
    return {}

def save_captions(captions):
    caption_file = "captions.json"
    with open(caption_file, "w") as f:
        json.dump(captions, f)

def generate_captions(image_path):
    client = OpenAI()  # Remove the api_key parameter as it will be read from environment
    
    with open(image_path, "rb") as image_file:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please provide a detailed description of this airport floor plan map, focusing on practical navigation directions. Describe locations in terms of walking directions (e.g., 'walk straight ahead', 'turn left/right') and distances from entry points or major intersections. Avoid using image-relative positions like 'top right' or 'bottom left'. Include information about shops, gates, and key locations, describing how to reach them from main entrances or central points."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
    return response.choices[0].message.content

def process_images():
    captions = load_captions()
    files_folder = "files"
    for filename in os.listdir(files_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            if filename not in captions:
                image_path = os.path.join(files_folder, filename)
                captions[filename] = generate_captions(image_path)
    save_captions(captions)
    return captions

if __name__ == "__main__":
    process_images()
