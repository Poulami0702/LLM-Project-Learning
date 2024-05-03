import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatWithMultiplePDF: 
    '''
    A class to handle chatting with multiple PDF documents
    
    '''
    def get_pdf_text(self,pdf_docs):
        '''
        Returns the text from a list of PDF documents
        '''

        print("*****get_pdf_text")
        text=""
        for pdf_doc in pdf_docs:
            reader = PdfReader(pdf_doc)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self,text):
        '''
        Returns a list of text chunks from the given text
        '''
        print("@@@@ get_text_chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks


    def get_vector_store(self,chunks):
        '''
        Returns a vector store from the given text chunks
        '''

        print("print API key : ",os.getenv("GEMINI_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain(self,vector_store):
        '''
        Returns a conversational chain from the given vector store
        '''
        print("**** get_conversational_chain")
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)   
        return chain  


    def user_input(self,user_question):
        '''
        Returns the answer to the given user question
        '''
        print("**** user_input")    
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
        vector_store = FAISS.load_local("faiss_index",embeddings, allow_dangerous_deserialization=True)
        docs=vector_store.similarity_search(user_question)

        chain = self.get_conversational_chain(vector_store)
        
        response = chain(
            {
                "input_documents": docs,
                "question": user_question,
            },
            return_only_outputs=True,
        )

        print(response)

        return response["output_text"]




