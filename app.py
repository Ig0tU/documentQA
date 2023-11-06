import streamlit as st
import langchain
import time
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS


llm = GooglePalm(google_api_key=st.secrets["api_key"], temperature=0.5)

st.title('Internet Documents Research Tool')
st.sidebar.title('Please put the URLs you want to chat with here:')
st.image('cover.jpg')

urls = []

for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

button = st.sidebar.button('Digest documents!')

mainPlace = st.empty()

if button:    
    mainPlace.subheader('Digesting the Documents...')
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',',','.'],
        chunk_size=1000,
        chunk_overlap=200
      )     

    docs = splitter.split_documents(data)
    
    mainPlace.subheader('Embedding and Creating Vector Database...')
    embeddings = GooglePalmEmbeddings(google_api_key=st.secrets["api_key"])
    vectorindex_googlepalm = FAISS.from_documents(docs, embeddings)
    vectorindex_googlepalm.save_local('vectordatabase')
    mainPlace.subheader('Vector Database Created.')
    

question = mainPlace.text_input('##### Now! Please type your prompt (question) here to find the answer in the documents: \n For example: what will the price of gold in 2024? please explain ')   

if question:
    embeddings = GooglePalmEmbeddings(google_api_key=st.secrets["api_key"])
    vectorindex_googlepalm = FAISS.load_local('vectordatabase',embeddings)
    chain = RetrievalQA.from_llm(llm=llm, retriever=vectorindex_googlepalm.as_retriever())
    answer = chain({'query':question},return_only_outputs=True)
    st.header('Answer: ')
    st.write(answer['result'])
