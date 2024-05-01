#vision API

from dotenv import load_dotenv

load_dotenv()

import streamlit as st 

import os
import pathlib
import textwrap
from PIL import Image


import google.generativeai as genai 

apikey = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=apikey)

##Function to load openAI 

def get_gemini_response(input,image):
    model=genai.GenerativeModel('gemini-pro-vision')
    if input!="":
        response=model.generate_content([input,image])
    else:
        response=model.generate_content(image)
    return response.text

#initialize our streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini application with image")

input=st.text_input("Input Prompt: ",key="input")

uploaded_file=st.file_uploader("choose an imae ....", type=["jpg","jpeg","png"])

image=""

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Describe image")

## if ask buttton is clicked 

if submit:
    response=get_gemini_response(input,image)
    st.subheader('The Response :')
    st.write(response)