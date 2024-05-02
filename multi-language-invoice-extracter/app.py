from dotenv import load_dotenv

load_dotenv()

import streamlit as st 
import os
from PIL import Image

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

## Function to load Gemini Pro Vision 
model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input,image,prompt):
    response=model.generate_content([input,image[0], prompt])
    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

## initialize our streamlit app

st.set_page_config(page_title="Multilanguage invoice Extracter")

st.header("Gemini Application for invoice extraction")

input=st.text_input("Input Prompt: ",key="input")
uploaded_file=st.file_uploader("choose an image of the invoice for extraction...", type=["jpg","jpeg","png"])

image=""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="uploaded Image.", use_column_width=True)

submit=st.button("Ask Me!!!!")

input_prompt = """
               You are an expert in understanding invoices.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """


if submit :
    image_data= input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_data, input)
    st.subheader("The response is")
    st.write(response)