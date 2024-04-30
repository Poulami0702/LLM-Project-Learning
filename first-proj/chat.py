#Q&A Chatbot

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

apikey=os.getenv("GEMINI_API_KEY")
genai.configure(api_key=apikey)

##Function to load OpenAI model and get responses 
model=genai.GenerativeModel('gemini-pro')
chat=model.start_chat(history=[])

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response

## initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Gemini Application - Q&A Demo")

input=st.text_input("Input: ", key="input")

submit=st.button("Ask me question")

## If ask button is clicked 

if submit:
    response=get_gemini_response(input)
    st.subheader("The Response is:")

    for chunk in response:
        print(st.write(chunk.text))
        print('_'*80)

    st.write(chat.history)