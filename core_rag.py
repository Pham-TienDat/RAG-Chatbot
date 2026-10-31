import os

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client

DB_FAISS_PATH = "vectorstore/db_faiss"
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return db


def build_rag_chain(vectorstore, k=4, temperature=0.2, max_tokens=1024):
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY,
    )
    client = Client()
    retrieval_qa_chat_prompt = client.pull_prompt(
        "langchain-ai/retrieval-qa-chat",
        dangerously_pull_public_prompt=True,
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": k}),
        combine_docs_chain,
    )
    return rag_chain


def ask(rag_chain, prompt: str):
    response = rag_chain.invoke({"input": prompt})
    result = response["answer"]
    source_documents = response["context"]
    return result, source_documents
