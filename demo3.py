import os
import streamlit as st
import io
import time
import re
import time
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import  FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pptGen import extract_challenges, generate_powerpoint_slide, extract_challenge_explanations, extract_solutions, extract_solution_explanations, upload_to_azure_blob, upload_to_dropbox, generate_sas_url, generate_and_upload_presentation, check_preview_download_buttons, create_url_link
from langchain.callbacks import get_openai_callback


#what do you think this will go next?
#meet with kathy and kris 
#change data structure
#make report for ryan benchmark and visa add years to blob storage
#


# Set page configuration
st.set_page_config(page_title="Chris Bot", page_icon="🤖")
filename = "testButtons.pptx"
ppt_container = "ppt-container"
os.environ['DROPBOX_TOKEN'] = 'sl.BiDyeLJ-kbesEp8LVHea60i1O1PsNawgO7Y3EBrcDtmU_LZbivK5Vl5n7jGDQc8lh44sJp7TKOiYSU8eEdw0sWVwgXslRN22osAidX4p8C6yNTmyV4MYg9v7K1wpBNQj5hQz9M3E'


# Set environment variables
os.environ["OPENAI_API_KEY"] ="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"
connection_string = "DefaultEndpointsProtocol=https;AccountName=hiaistorage;AccountKey=Ou24UdNkdIVikn99RUiuWURdOdZ15/sFWiuZA0Pbof3mqDeRlwzPgnSLmaMJtb05xad2qzpB6O/++AStHscqiQ==;EndpointSuffix=core.windows.net"

def process_Company(company):
    docs = get_docs(company)
    text = get_pdf_text(docs)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation=get_conversation(vectorstore)
    return st.session_state.conversation

# Function to get documents from Azure blob storage
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
            if container_name != "insightsolutions":
                st.write(blob.name)
    return pdf_docs

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=4000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get vectorstore from text
def get_vectorstore(text):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)
    return vectorstore

# Function to get conversation from vectorstore
def get_conversation(vectorstore):
    llm = ChatOpenAI(temperature=.4,model="gpt-3.5-turbo-16k")
    #prompt_template = PromptTemplate.from_template("When you are asked to find three solutions, provide the case studies you mentioned.The format is: Case Study:, Challenge:, Solution:")
    memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        memory=memory)
     #   condense_question_prompt=prompt_template)
    return conversationChain

# Function to simulate typing effect
def simulate_typing_effect(message, message_placeholder):
    for i in range(len(message)):
        message_placeholder.markdown(message[:i+1] + "▌")
        time.sleep(0.0075)  #adjust this value to change the speed of the typing effect
    message_placeholder.markdown(message)


def extract_challenges(most_recent_chat_history):
    content = ""
    for message in most_recent_chat_history:
        if message['role'] == 'assistant':
            content += message['content'] + "\n"
    matches = re.findall(r'\d+\.\s+(.*?)(:|$)', content) # 1. ______ :
    challenges = [match[0].strip() for match in matches[:3]]  # Get up to the first 3 challenges

    return challenges

def find_solutions_button(challenges):
    # Function to check generate ppt button
    if len(challenges) >= 3:
        if st.button("Match to previous client stories", key="solutions"):
            return True
    return False

def cleanchat(chatHist):

    content = chatHist[0]['content']
    return content

def check_generate_ppt_button(challenges, solutions):
    # st.write("inside check gen ppt")
    # st.write("len of challenges")
    # st.write(len(challenges))
    # st.write("len og solutions")
    # st.write(len(solutions))
    if len(challenges) >= 3 and len(solutions) >= 3:
        if st.button("Generate PowerPoint"):
            st.write("You clicked the button")
            time.sleep(1)
            return True
    return False

# Main function
def main():
    st.title("Sales Assistant Engine")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # if "challenge_button" not in st.session_state:
    #     st.session_state.
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    running_total_cost = 0
    running_total_placeholder = st.sidebar.empty()
    st.sidebar.markdown("### Here are some prompts you can ask me!")
    st.sidebar.markdown("- What are three challenges being faced?")
    st.sidebar.markdown("- What are some possible business opportunities")
    st.sidebar.markdown("- What are some risks being faced?")
    st.sidebar.markdown("- Can you summarize results from operations")
    st.sidebar.markdown("- How much revenue was made?")
    st.sidebar.markdown("- What are some risks being faced?")
    st.sidebar.markdown("### Here are some other ")

    # Create dropdown menu for company selection
    records = ['','Carnival','Axon','Gov','Tesla', 'Starbucks', 'Toyota', 'InsightSolutions','Bridgestone',"Progressive" ,'Benchmark', 'Activision', 'Avnet', 'Denso', 'Fidelity', 'L3Harris', 'Markel', 'McDonalds', 'Nvidia', 'Microchip', 'SAP', 'UHaul', 'Stanford', 'Visa']    
    company = st.selectbox("What company would you like to research?", records)

    
    # Create search button
    search_button_clicked = False
    if not search_button_clicked:
        if st.button("Search", key="searchbutton"):
            with st.spinner("Processing financial docs"):
                st.session_state.conversation = process_Company(company)
                search_button_clicked = True

    # Create chat input for user question
    if prompt := st.chat_input("Ask a question about this company"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response with typing effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with get_openai_callback() as cb:
                response = st.session_state.conversation({'question':prompt})

            assistant_response = response['chat_history'][-1].content
            st.session_state.messages.append({"role": "assistant", "content":assistant_response})
            simulate_typing_effect(response['chat_history'][-1].content, message_placeholder)
            running_total_cost += cb.total_cost
            st.sidebar.text(f"Running Total Cost (USD): ${running_total_cost:.5f}")

    most_recent_chat_history= st.session_state.messages[-1:]
    challenges = extract_challenges(most_recent_chat_history)
    
    solutions= []
    if find_solutions_button(challenges):
        company2 = "InsightSolutions"
        st.write("Gathering Insight Client stories. This may take a while...")
        st.session_state.conversation=process_Company(company2)
        chatHist = cleanchat(most_recent_chat_history)

        user_greeting = f"What are some solutions to these challenges? Provide references to what Insight did to help. REFENCE A CLIENT STORY WHEN POSSIBLE. These are the challenges: {chatHist} "
        st.session_state.messages.append({"role": "user", "content": user_greeting})
        with st.chat_message("user"):
            st.markdown(user_greeting)
        
    # Generate and display response with typing effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with get_openai_callback() as cb:
                response = st.session_state.conversation({'question': user_greeting})
            st.session_state.messages.append({"role": "assistant", "content": response['chat_history'][-1].content})
            simulate_typing_effect(response['chat_history'][-1].content, message_placeholder)
            running_total_cost += cb.total_cost
            st.sidebar.text(f"Running Total Cost (USD): ${running_total_cost:.5f}")

        #need to see if solutions were generated in order to proceed. 
        most_recent_chat_history= st.session_state.messages[-1:]

        content = ""
        for message in most_recent_chat_history:
            if message['role'] == 'assistant':
                content += message['content'] + "\n"
                matches = re.findall(r'\d+\.\s+(.*?)(:|$)', content) # 1. ______ :
                solutions = [match[0].strip() for match in matches[:3]]  # Get up to the first 3 solutions

    if check_generate_ppt_button(challenges, solutions):    
        st.write("Generating and uploading presentation...")
        # debug message
        # download_link, preview_link = generate_and_upload_presentation(most_recent_chat_history, filename, company)
        # st.write("Presentation generated and uploaded!")  # debug message
        # check_preview_download_buttons(download_link, preview_link)
    
    

# Run main function

if __name__ == '__main__':
     main()