from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## using Wikipedia tools
print("### using Wikipedia tools")
api_wrapper=WikipediaAPIWrapper(top_k_results=1,
                                doc_content_chars_max=2000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
print("Name of the tool : ", wiki_tool.name)


##using webbaseloader 
print("### using webbaseloader")
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500).split_documents(docs)   
# vector_store=FAISS.from_documents(documents,OllamaEmbeddings()) 
vector_store = Chroma.from_documents(documents,OllamaEmbeddings())
retriever=vector_store.as_retriever()
print("retriever : - ", retriever)


## calling langchain hub for prompt - rlm/rag-prompt
print("### calling langchain hub for prompt - rlm/rag-prompt")
# prompt = hub.pull("rlm/rag-prompt")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
print("### Promt Message : using model - rlm/rag-prompt - from langchainhub",prompt.messages)


## retrive document using retriever tool
print("### retrive document using retriever tool")
retriever_tool=create_retriever_tool(retriever, "langsmith_search_tools",
                                      "Search for information about LangSmith.")

print("### retriever tool : ", retriever_tool.name)


### Arxiv tool using Arxiv API
print("### Arxiv tool using Arxiv API")
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,
                                doc_content_chars_max=2000)

arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper) 

print("$$ Arxiv tools name : ", arxiv_tool.name)


## adjusting tools
print("### adjusting tools")
tools = [wiki_tool, retriever_tool, arxiv_tool] 
print("### all the tools together : ", tools)

## Ollama2 LLM
print("### Ollama2 LLM")
llm=Ollama(model="llama2",
        temperature=0.7,
           )

output_parser=StrOutputParser()

chain = prompt|llm|output_parser

print("### chain : ", chain)
result=chain.invoke({
                     "question":"Tell me about India"})

print("### result : ", result)