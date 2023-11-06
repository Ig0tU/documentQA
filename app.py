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
#st.subheader(' ')

urls = []

for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

button = st.sidebar.button('Digest documents!')

mainPlace = st.empty()

if button:    
    mainPlace.text('Digesting the Documents...')
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=200
)
    docs = splitter.split_documents(data)
    
    mainPlace.text('Embedding and Creating Vector Database...')
    embeddings = GooglePalmEmbeddings(google_api_key=st.secrets["api_key"])
    vectorindex_googlepalm = FAISS.from_documents(docs, embeddings)
    chain = RetrievalQA.from_llm(llm=llm, retriever=vectorindex_googlepalm.as_retriever())
    
    mainPlace.text('Vector Database Created.')
    

question = mainPlace.text_input('Please type your question here: ')   

if question:
    answer = chain({'query':question},return_only_outputs=True)

    st.header('Answer: ')
    st.subheader(answer['result'])
