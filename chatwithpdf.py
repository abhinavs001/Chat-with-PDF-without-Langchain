import json
import os
import streamlit as st
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPAuthorizationCredentials
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.middleware.cors import CORSMiddleware
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext
from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.vector_stores.cassandra import CassandraVectorStore
from copy import deepcopy
from tempfile import NamedTemporaryFile
import uvicorn
import gradio as gr
from pydantic import BaseModel

#database connection
@st.cache_resource
def create_datastax_connection():

    cloud_config= {
        'secure_connect_bundle': 'secure-connect-checkdatabase.zip'
    }

    with open("token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session
#main function
def main():

    index_placeholder = None
    st.set_page_config(page_title = "Chat with PDF using Llama2 & Llama Index", page_icon="🦙")
    st.header('Chat with PDF using Llama2 model & Llama Index')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    session = create_datastax_connection()
    os.environ['GRADIENT_ACCESS_TOKEN'] = ''
    os.environ['GRADIENT_WORKSPACE_ID'] = ''
#choosing LLM model from open source models available
    llm = GradientBaseModelLLM(
        base_model_slug="llama2-7b-chat",
        max_tokens=400,
    )
#model embedding
    embed_model = GradientEmbedding(
        gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large",
    )

    service_context = ServiceContext.from_defaults(
        llm = llm,
        embed_model = embed_model,
        chunk_size=256,
    )

    set_global_service_context(service_context)
#streamlit environment setup and ready interface
    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF',
                                accept_multiple_files = False, 
                                type=['pdf'])
        if st.button('Process'):
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(docs.getbuffer())
                with st.spinner('Processing'):
                    documents = SimpleDirectoryReader(".").load_data()
                    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                    query_engine = index.as_query_engine()
                    if "query_engine" not in st.session_state:
                        st.session_state.query_engine = query_engine
                    st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Message your PDF!"):
            with st.chat_message("User"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "User","content": prompt})
            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            cleaned_response = pdf_response.response
            with st.chat_message("Assistant"):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "Assistant", "content": cleaned_response})
        else:
            st.markdown(
                'Upload PDFs to chat'
                )

if __name__ == '__main__':
    main()