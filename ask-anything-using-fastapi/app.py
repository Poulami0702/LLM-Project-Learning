from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn 
import os

from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()


app=FastAPI(
    title="LangChain Server",
    description="A simple API Server",
    version="0.0.1 "
)

llm=Ollama(model_name="llama2")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(app,prompt1|llm,path="/essay")

add_routes(app,prompt2|llm,path="/poem")


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8051)