from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st

from langchain import LLMBashChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def extract_text_from_pdf(pdf_reader):
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def build_knowledge_base(text):
    """Splits text into chunks and converts each chunk
    into a vector using OpenAI embeddings, then builds
    the semantic index using the Facebook FAISS similarity search"""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base


def prompt_user_for_question():
    user_question = st.text_input("Ask a question about your PDF:")
    return user_question


def retrieve_results(user_question, knowledge_base):
    """Takes the user question and relevant embeddings and
    passes them to the OpenAI llm and returns back a summary"""
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question, chain=chain)
        # Log token usage to console
        print(cb)
    return response


def main():
    load_dotenv()

    # Configure Streamlit page
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    # Upload the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Load the file contents
        pdf_reader = PdfReader(pdf)

        # Extract the text
        text = extract_text_from_pdf(pdf_reader)

        # Build the knowledge base
        knowledge_base = build_knowledge_base(text)

        # Prompt user for query
        user_question = prompt_user_for_question()

        # Retrieve the relevant embeddings and pass it to llm
        if user_question:
            response = retrieve_results(user_question, knowledge_base)
            st.write(response)


if __name__ == "__main__":
    main()
