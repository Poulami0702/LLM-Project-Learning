from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


## using Wikipedia tools
api_wrapper=WikipediaAPIWrapper(top_k_results=1,
                                doc_content_chars_max=2000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
print("Name of the tool : ", wiki_tool.name)
# print(wiki_tool.run("What is the capital of India"))

##using webbaseloader 

loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500).split_documents(docs)   
vector_store=FAISS.from_documents(documents,OllamaEmbeddings()) 
retriever=vector_store.as_retriever()
print("retriever", retriever)
