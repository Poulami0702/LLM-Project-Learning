import requests
import streamlit as st

def get_essay_response(input_text):
    response=requests.post("http://localhost:8501/essay/invoke",
                           json={'input':{'topic':input_text}})
    
    return response.json()['output']


def get_poem_response(input_text):
    response=requests.post("http://localhost:8501/poem/invoke",
                           json={'input':{'topic':input_text}})

    return response.json()['output']


## streamlit framework
st.title("Generate essay or poem - using LLAMA2 API")
# Define dropdown options
options = ["Essay", "Poem"]

# Create the dropdown
selected_option = st.selectbox("Select an option:", options)
print("selected option : ", selected_option)
input_text=st.text_input("Enter topic")

if input_text and selected_option:
    if selected_option == "Essay":
        response = get_essay_response(input_text)
    else:
        response = get_poem_response(input_text)

    st.write(response)