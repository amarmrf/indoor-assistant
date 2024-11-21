import streamlit as st
import os
import json
from dotenv import load_dotenv
from PIL import Image
import base64
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Function to generate captions for images using Mistral Pixtral
def generate_captions(image_path):
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Please provide a detailed description of this airport floor plan map, including information about shops, gates, and key locations."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
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
            print(f"Processing image: {filename}")  # Add this line
            if filename not in captions:
                image_path = os.path.join(files_folder, filename)
                captions[filename] = generate_captions(image_path)
    save_captions(captions)
    return captions

def get_mistral_response(user_question, context, image_paths):
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": f"""
            You are an expert airport navigation assistant. Your task is to provide clear, step-by-step directions 
            for navigating within the airport. Use the provided floor plans and/or maps as well as the given context prompt to give accurate and helpful instructions.
            
            Look at how to move from one point to another in the airport and provide instructions on how to do so.
            Context about the airport layout:
            {context}

            When providing navigation instructions, follow these guidelines:
            1. Identify the start and end points from the user's question.
            2. Plan the most efficient route.
            3. Provide clear, concise directions using landmarks and key locations.
            4. Use simple, conversational language.
            5. Include specific details and easily identifiable signs.
            6. Anticipate potential questions or issues.
            7. Include estimated walking time.

            Format your response as follows:
            Start point: [Identified start point]
            End point: [Identified end point]

            Route: [Brief overview of the route]

            Step-by-step directions:
            1. [First direction]
            2. [Second direction]
            3. [Third direction]
            ...

            Estimated time: [Estimated walking time]

            Additional info: [Any helpful tips or information]

            Here are two examples of well-formatted, human-like instructions:

            Example 1:
            Start point: Main Entrance
            End point: Gate B5

            Route: Main Entrance → Security Checkpoint → Central Concourse → B Concourse → Gate B5

            Step-by-step directions:
            1. From the Main Entrance, walk straight ahead towards the large "Security Checkpoint" sign.
            2. Go through security screening (have your boarding pass and ID ready).
            3. After security, follow signs to the "Central Concourse" and walk straight for about 2 minutes.
            4. At the Central Concourse, turn right and look for signs pointing to "B Concourse".
            5. Walk for about 3 minutes until you reach the entrance to B Concourse.
            6. Enter B Concourse and continue walking, keeping an eye on the gate numbers.
            7. Gate B5 will be on your right side, about halfway down the concourse.

            Estimated time: 10-12 minutes

            Example 2:
            Start point: Baggage Claim Area
            End point: Airport Shuttle Pick-up

            Route: Baggage Claim Area → Ground Transportation Level → Shuttle Pick-up Zone

            Step-by-step directions:
            1. From the Baggage Claim Area, locate the escalators or elevators near carousel 3.
            2. Take the escalator or elevator down one level to the "Ground Transportation" area.
            3. Once on the Ground Transportation level, look for signs pointing to "Shuttle Pick-up".
            4. Walk straight ahead, passing the car rental counters on your left.
            5. Exit the building through the automatic doors marked "Ground Transportation".
            6. The Shuttle Pick-up area will be directly in front of you, with clearly marked lanes for different shuttles.
            7. Look for your specific shuttle (hotel, parking, or car rental) and wait in the designated area.

            Estimated time: 5-7 minutes

            Additional info: Shuttles typically run every 10-15 minutes. If you don't see your shuttle, wait in the designated area, and one should arrive shortly.

            If the user's question doesn't ask for directions or doesn't provide enough information, 
            politely ask for clarification instead of providing directions.
        """},
        {"role": "user", "content": [
            {"type": "text", "text": f"Based on this user question, provide navigation instructions: '{user_question}'"}
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
        model="pixtral-12b-2409",
        messages=messages
    )

    return chat_response.choices[0].message.content


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

    # User input
    user_input = st.text_input("Ask a question about the airport:")

    if user_input:
        image_paths = [os.path.join("files", filename) for filename in captions.keys()]
        response = get_mistral_response(user_input, context, image_paths)
        st.write("Assistant:", response)
        st.write("Using Mistral model: pixtral-12b-2409")

    # Display images
    st.subheader("Airport Map")
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
