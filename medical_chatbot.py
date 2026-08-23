import os

import streamlit as st
from dotenv import load_dotenv


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# from langchain_classic.chains import RetrievalQA
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


# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(
#         template=custom_prompt_template, input_variables=["context", "question"]
#     )
#     return prompt


def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # custom_prompt_template = """

        #     Context: {context}
        #     Question: {question}
        #     """
        # huggingface_repo_id = "Qwen/Qwen2.5-7B-Instruct"

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=1024,
                api_key=GROQ_API_KEY,
            )

            # Buld RAG Chain
            client = Client()
            retrieval_qa_chat_prompt = client.pull_prompt(
                "langchain-ai/retrieval-qa-chat", dangerously_pull_public_prompt=True
            )
            combine_docs_chain = create_stuff_documents_chain(
                llm, retrieval_qa_chat_prompt
            )
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
            )

            response = rag_chain.invoke({"input": prompt})
            result = response["answer"]
            source_documents = response["context"]
            result_to_show = result
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append(
                {"role": "assistant", "content": result_to_show}
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
