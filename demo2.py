import os
import streamlit as st
import io
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import  FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template



os.environ["OPENAI_API_KEY"] ="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"
connection_string = "DefaultEndpointsProtocol=https;AccountName=hiaistorage;AccountKey=Ou24UdNkdIVikn99RUiuWURdOdZ15/sFWiuZA0Pbof3mqDeRlwzPgnSLmaMJtb05xad2qzpB6O/++AStHscqiQ==;EndpointSuffix=core.windows.net"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_docs(container_name):

    container_name = container_name.lower()
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)


    container_client = blob_service_client.get_container_client(container_name)

    pdf_docs = []

    for blob in container_client.list_blobs():
        if blob.name.endswith('.pdf'):
            blob_client = container_client.get_blob_client(blob.name)
            stream = blob_client.download_blob().readall()
            pdf_docs.append(io.BytesIO(stream))
            st.write(blob.name)
    return pdf_docs


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)
    return vectorstore


def get_conversation(vectorstore):
    llm = ChatOpenAI()   
    memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )
    return conversationChain

def handle_userinput(userQuestion):
    response= st.session_state.conversation({'question':userQuestion})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2  == 0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html= True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html= True)


def main():
    st.set_page_config(page_title="RecApp", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    #streamlit reloads code when a button is pressed and doing this prevents that from happening. 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None 


    st.header("RecApp :books:")
    records = ['','Carnival','Axon','Gov','Tesla']

    company = st.selectbox("What company would you like to research?:", records)
    
    # st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "HELLO HUMAN"),unsafe_allow_html=True)
    if st.button("Search"):
        with st.spinner("Processing financial docs"):

            #get docs from azure blob
            docs = get_docs(company)

            #get text from documents
            text = get_pdf_text(docs)

            #get chunks from text
            text_chunks = get_text_chunks(text)
            
            #create vector store. This is created temporarly. We need to save them to a vector database to access it later. Otherwise we need to make new embeddings everytime
            vectorstore = get_vectorstore(text_chunks)

            #create conversation chain
            st.session_state.conversation = get_conversation(vectorstore)
    
    userQuestion = st.text_input("Ask a question about this company")
    if userQuestion:
        handle_userinput(userQuestion)



if __name__ == '__main__':
     main()