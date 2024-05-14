## Text Loader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
import numpy as np
import ollama
import chromadb

loader=TextLoader("data/scripts1.txt")
text_documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
documents=text_splitter.split_documents(text_documents)
print(len(documents))
print("***text-splitte", documents[:5])


db = Chroma.from_documents(documents,OllamaEmbeddings())
query = "openAI?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)