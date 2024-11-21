import streamlit as st
import os
import json
from dotenv import load_dotenv
import openai
from PIL import Image
import ollama
import base64  # Add this import
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate captions for images using GPT-4
def generate_captions(image_path):
    with open(image_path, "rb") as image_file:
        client = openai.OpenAI()  # Create a client instance
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please provide a detailed description of this airport floor plan map, including information about shops, gates, and key locations."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                    ],
                }
            ],
            max_tokens=500
        )
    return response.choices[0].message.content

# Function to load captions from file
def load_captions():
    caption_file = "captions.json"
    if os.path.exists(caption_file):
        with open(caption_file, "r") as f:
            return json.load(f)
    return {}

# Function to save captions to file
def save_captions(captions):
    caption_file = "captions.json"
    with open(caption_file, "w") as f:
        json.dump(captions, f)

# Function to process images and generate captions
def process_images():
    captions = load_captions()
    files_folder = "files"
    for filename in os.listdir(files_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename not in captions:
                image_path = os.path.join(files_folder, filename)
                captions[filename] = generate_captions(image_path)
    save_captions(captions)
    return captions

# Function to get response from OpenAI GPT
def get_openai_response(prompt, context, image_paths):
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": [
            {"type": "text", "text": f"""
                Please answer the following question about the airport using a step-by-step thought process:

                {prompt}

                1. First, consider the relevant information from the airport floor plans.
                2. Then, think about how this information relates to the question.
                3. Finally, provide a clear and concise answer.

                Please format your response as follows:
                Thought process:
                1. [Your first thought]
                2. [Your second thought]
                3. [Your third thought]

                Answer: [Your final answer based on the thought process]
            """}
        ]}
    ]
    
    # Add images to the message
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content

def get_mistral_response(prompt, context, model_name, image_paths):
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": [
            {"type": "text", "text": f"""
                Please answer the following question about the airport using a step-by-step thought process:

                {prompt}

                1. First, consider the relevant information from the airport floor plans.
                2. Then, think about how this information relates to the question.
                3. Finally, provide a clear and concise answer.

                Please format your response as follows:
                Thought process:
                1. [Your first thought]
                2. [Your second thought]
                3. [Your third thought]

                Answer: [Your final answer based on the thought process]
            """}
        ]}
    ]

    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    chat_response = client.chat.complete(
        model=model_name,
        messages=messages
    )

    return chat_response.choices[0].message.content

# Function to get response from Ollama LLaVA
def get_ollama_response(prompt, context, model_name, image_paths):
    image_paths_str = "\n".join(image_paths)
    full_prompt = f"""{context}
            User: Please answer the following question about the airport using a step-by-step thought process:

            {prompt}

            1. First, consider the relevant information from the airport floor plans.
            2. Then, think about how this information relates to the question.
            3. Finally, provide a clear and concise answer.

            Please format your response as follows:
            Thought process:
            1. [Your first thought]
            2. [Your second thought]
            3. [Your third thought]

            Answer: [Your final answer based on the thought process]

            The following image paths are available for reference:
            {image_paths_str}
            """
    
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': full_prompt}]
    )
    return response['message']['content']

# Initialize captions
@st.cache_resource
def initialize_captions():
    return load_captions()

# Streamlit app
def main():
    st.title("Airport Navigation Assistant")

    # Load pre-generated captions
    captions = initialize_captions()
    context = "\n\n".join(captions.values())

    # Model selection
    model = st.radio("Select Model", ("OpenAI GPT", "Ollama LLaVA", "Mistral AI"))

    # User input
    user_input = st.text_input("Ask a question about the airport:")

    ollama_model = None
    mistral_model = "pixtral-12b-2409"

    if model == "Ollama LLaVA":
        ollama_model = st.selectbox("Select Ollama Model", ["llava:13b", "llava:7b", "llava:34b"])

    if user_input:
        image_paths = [os.path.join("files", filename) for filename in captions.keys()]
        if model == "OpenAI GPT":
            response = get_openai_response(user_input, context, image_paths)
        elif model == "Ollama LLaVA":
            response = get_ollama_response(user_input, context, ollama_model, image_paths)
        else:  # Mistral AI
            response = get_mistral_response(user_input, context, mistral_model, image_paths)
        
        st.write("Assistant:", response)
        
        # Display the chosen model
        if model == "Ollama LLaVA":
            st.write(f"Using Ollama model: {ollama_model}")
        elif model == "Mistral AI":
            st.write(f"Using Mistral model: {mistral_model}")


    # Display images
    st.subheader("Airport Floor Plans")
    for filename, caption in captions.items():
        image_path = os.path.join("files", filename)
        image = Image.open(image_path)
        st.image(image, caption=filename)
        # Remove this line as we're now displaying captions separately
        # st.write(caption)

        # Display generated captions
    st.subheader("Generated Captions")
    for filename, caption in captions.items():
        st.write(f"**{filename}:**")
        st.write(caption)
        st.write("---")

if __name__ == "__main__":
    main()