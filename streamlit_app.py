from collections import namedtuple
import streamlit as st
from langchain.llms import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQA
import PyPDF2

st.set_page_config(page_title="Ask the assistant")
st.title("I want an assistant")

def generate_response(uploaded_file, openai_api_key, query_text):
    """
        response from the uploaded file
    """
    if uploaded_file is not None:
        documents = [PyPDF2.PdfReader(uploaded_file).pages[0].extract_text()]

    # Split documents into text chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)

    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = chroma.from_documents(texts, embeddings)

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=openai(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)

    return qa.run(query_text)


# File upload
uploaded_file = st.file_uploader('Upload an article', type=['pdf'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

openai_api_key = st.secrets["OPENAI_API_KEY"]

result = []

with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)

if len(result):
    st.info(response)