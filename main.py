import streamlit as st
import os
import json
from dotenv import load_dotenv
import openai
from PIL import Image
import ollama
import base64
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load navigation data
def load_navigation_data():
    with open("nav.txt", "r") as f:
        return f.read()

# Function to get response from OpenAI GPT
def get_openai_response(prompt, context):
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": "You are an airport navigation assistant. Use the provided navigation data to give clear, step-by-step directions."},
        {"role": "user", "content": f"""
            Using this navigation data:

            {context}

            Please answer the following question:
            {prompt}

            Provide step-by-step directions in a clear, concise format.
        """}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content

def get_mistral_response(prompt, context, model_name):
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are an airport navigation assistant. Use the provided navigation data to give clear, step-by-step directions."},
        {"role": "user", "content": f"""
            Using this navigation data:

            {context}

            Please answer the following question:
            {prompt}

            Provide step-by-step directions in a clear, concise format.
        """}
    ]

    chat_response = client.chat.complete(
        model=model_name,
        messages=messages
    )

    return chat_response.choices[0].message.content

def get_ollama_response(prompt, context, model_name):
    full_prompt = f"""You are an airport navigation assistant. Use the provided navigation data to give clear, step-by-step directions.

    Using this navigation data:

    {context}

    Please answer the following question:
    {prompt}

    Provide step-by-step directions in a clear, concise format.
    """
    
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': full_prompt}]
    )
    return response['message']['content']

# Streamlit app
def main():
    st.title("Airport Navigation Assistant")

    # Load navigation data
    nav_context = load_navigation_data()

    # Model selection
    model = st.radio("Select Model", ("OpenAI GPT", "Ollama", "Mistral AI"))

    # User input
    user_input = st.text_input("Where would you like to go in the airport?")

    ollama_model = None
    mistral_model = "mistral-medium"

    if model == "Ollama":
        ollama_model = st.selectbox("Select Ollama Model", ["llama2", "mistral", "neural-chat"])

    if user_input:
        if model == "OpenAI GPT":
            response = get_openai_response(user_input, nav_context)
        elif model == "Ollama":
            response = get_ollama_response(user_input, nav_context, ollama_model)
        else:  # Mistral AI
            response = get_mistral_response(user_input, nav_context, mistral_model)
        
        st.write("Directions:", response)
        
        # Display the chosen model
        if model == "Ollama":
            st.write(f"Using Ollama model: {ollama_model}")
        elif model == "Mistral AI":
            st.write(f"Using Mistral model: {mistral_model}")

    # Display navigation data sections
    st.subheader("Available Navigation Routes")
    sections = nav_context.split("##")
    for section in sections[1:]:  # Skip the first empty section
        section_title = section.split("\n")[0].strip()
        st.write(f"**{section_title}**")
        with st.expander("Show routes"):
            st.write(section)

if __name__ == "__main__":
    main()