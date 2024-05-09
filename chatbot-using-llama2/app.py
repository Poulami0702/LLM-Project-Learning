from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## Streamlit frame work

st.title("Ask AI")

input_question = st.text_input("Search the topic you want")

## Ollama2 LLM

llm=Ollama(model="llama2",
           temperature=0.7
           )

output_parser=StrOutputParser()

chain = prompt|llm|output_parser

if input_question:
    result=chain.invoke({"question":input_question})
    st.write(result)