import streamlit as st
import langchain
import time
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS


llm = GooglePalm(google_api_key=st.secrets["api_key"], temperature=0.5, max_tokens=300)

st.write(':red[If this page returns an error, it is probably because the Google generative ai API key is unavailable where the app is running.]')
st.title('QA Chatbot')
st.image('cover.jpg')
st.subheader('Please enter your question here about our courses:.')

with st.spinner('Loading, please wait...'):
  loader = CSVLoader(file_path='QA.csv', source_column= 'prompt',encoding='cp1252')
  data = loader.load()
  embeddings = GooglePalmEmbeddings(google_api_key=st.secrets["api_key"])
  vectorindex_googlepalm = FAISS.from_documents(data, embeddings)
  retriever = vectorindex_googlepalm.as_retriever()
  vectorindex_googlepalm.save_local('vectordatabase')

question = st.text_input('enter your question here:')

if question: 
    vectorindex_googlepalm = FAISS.load_local('vectordatabase',embeddings)

    retriever = vectorindex_googlepalm.as_retriever(score_threshold=0.7)

    chain = RetrievalQA.from_chain_type(llm= llm,
                              chain_type='stuff',
                              retriever = retriever,
                              input_key='query',
                              return_source_documents=True)

    answer = chain(question)


    st.header('Assistant:')
    st.subheader(answer['result'])