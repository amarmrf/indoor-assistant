import streamlit as st
import os
import json
from dotenv import load_dotenv
from PIL import Image
import base64
from mistralai import Mistral  # Replace OpenAI import



# Load environment variables
load_dotenv()

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


def get_response(user_question, context, image_paths):
    api_key = os.getenv("MISTRAL_API_KEY")  # Change to MISTRAL_API_KEY
    client = Mistral(api_key=api_key)  # Use Mistral client

    # Split analysis questions into separate prompts
    location_prompt = {"role": "system", "content": f"""
        You are an expert airport navigation assistant. Before I let you help to provide clear, step-by-step directions 
        for navigating within the airport. You need to analyze the location aspects of this navigation request.
        
        1. Location Analysis:
        - What is the starting point mentioned in the request?
        - What is the destination point?
        - Are both points clearly specified? If not, what clarification is needed?
        
        Base your analysis on this airport layout context:
        {context}
    """}

    landmark_prompt = {"role": "system", "content": f"""
        Analyze the landmarks relevant to this navigation request:
        
        2. Landmark Analysis:
        - What are the major landmarks visible near the starting point?
        - What are the major landmarks visible near the destination?
        - What significant landmarks exist along potential routes?
    """}

    route_prompt = {"role": "system", "content": f"""
        Analyze the possible routes for this navigation request:
        
        3. Route Planning:
        - What are the possible routes between these points?
        - Which route is most efficient considering distance and ease of navigation?
        - What potential obstacles or busy areas should be considered?

    """}

    navigation_considerations_prompt = {"role": "system", "content": f"""
       Analyze the practical navigation considerations:
        
        4. Navigation Considerations:
        - Are there any level changes (escalators, elevators) needed?
        - Are there any security checkpoints to consider?
        - Are there any time-sensitive factors to consider?

        When providing navigation instructions, follow these guidelines:
        1. Identify the start and end points from the user's question.
        2. Plan the most efficient route.
        3. Provide clear, concise directions using landmarks and key locations.
        4. Use simple, conversational language.

    """}

    def get_response_with_images(prompt, question):
        messages = [
            prompt,
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        
        # Add images to the messages
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        
        return client.chat.complete(  # Change to Mistral's API format
            model="pixtral-12b-2409",  # Use Pixtral model
            messages=messages
        ).choices[0].message.content

    # Get responses for each analysis aspect
    location_analysis = get_response_with_images(location_prompt, user_question)
    landmark_analysis = get_response_with_images(landmark_prompt, location_analysis)
    route_analysis = get_response_with_images(route_prompt, landmark_analysis)
    navigation_analysis = get_response_with_images(navigation_considerations_prompt, route_analysis)

    # Combine analyses
    combined_analysis = f"""
    Location Analysis:
    {location_analysis}

    Landmark Analysis:
    {landmark_analysis}

    Route Analysis:
    {route_analysis}

    Navigation Considerations:
    {navigation_analysis}
    """

    # Final navigation instructions prompt
    navigation_prompt = {"role": "system", "content": f"""
        Based on this analysis:
        {combined_analysis}

        Now you are ready to use the provided floor plans and/or maps as well as the given context prompt to give accurate and helpful instructions.
        
        Remember, annotations are only for helpers, don't use the helpers in the actual instructions. 
        For example food and beverage list may be scattered around the airport, so there might be no whole section with food and beverage list. This could be the case for other sections as well. 
        So give only relevant and actionable navigation instructions.
        
        Provide clear step-by-step navigation instructions following this format:

        Start point: [Identified start point]
        End point: [Identified end point]

        Route Overview: [Brief overview of the chosen route]
    
        Estimated time: [Estimated walking time]

        Additional info: [Any helpful tips or information]

        Step-by-step directions:
        1. [First direction]
        2. [Second direction]
        ...

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


    """}

    # Get final navigation instructions
    final_instructions = get_response_with_images(navigation_prompt, user_question)

    # Return combined response
    return f"""
    Analysis of Navigation Request:
    {combined_analysis}

    Navigation Instructions:
    {final_instructions}
    """

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

    # In the main() function, add:
    show_analysis = st.checkbox("Show detailed analysis", value=False)

    if user_input:
        image_paths = [os.path.join("files", filename) for filename in captions.keys()]
        response = get_response(user_input, context, image_paths)
        if show_analysis:
            st.write("Assistant:", response)
        else:
            # Extract only the navigation instructions part
            navigation_part = response.split("Navigation Instructions:")[1]
            st.write("Assistant:", navigation_part)

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
