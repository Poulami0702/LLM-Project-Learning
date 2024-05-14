
## Text Loader
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import bs4
import numpy as np
import ollama
import chromadb

loader=TextLoader("data/scripts1.txt")
text_documents=loader.load()

## load chunk and index the content of the html page 

# loader_web=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#                      bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#                          class_=("post-title","post-content","post-header")

#                      )))

# text_documents_web=loader_web.load()
# print("text_documents", text_documents_web)


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
documents=text_splitter.split_documents(text_documents)
print(len(documents))
print("***text-splitte", documents[:5])


# ollama.embeddings(
#   model='nomic-embed-text',
#   prompt='describe the text',
# )


client = chromadb.Client()
collection = client.create_collection(name="docs")
# Store each document in a vector embedding database
for i, doc in enumerate(documents):
    print("###### i", i)
    print("###### d", doc)
    
    # Extracting page content from the document
    page_content = doc.page_content if hasattr(doc, 'page_content') else ""
    
    # Extracting text from metadata if available
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    text_metadata = " ".join([str(value) for value in metadata.values()])
    
    # Combining page content and metadata text
    text_prompt = f"{page_content} {text_metadata}"
    
    response = ollama.embeddings(model="nomic-embed-text", prompt=text_prompt)
    embedding = response["embedding"]
    
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[{"text": text_prompt}]
    )



# an example prompt
prompt = "openAI ?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)

print("calling - response", response)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)

print("calling - results", results)
data = results['documents'][0][0]

print("calling - data", data)