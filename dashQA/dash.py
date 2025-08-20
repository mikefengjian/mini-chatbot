import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Path to the PDF file we want to use as our knowledge base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "dashQA.pdf")

# --- Streamlit UI setup ---
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.header("Ask questions about Dash 📄")

@st.cache_resource
def load_vectorstore():
    """
    Load the PDF, process it into chunks, embed the text,
    and create a Chroma vector database.
    Using @st.cache_resource ensures this heavy process runs only once.
    """
    # Load the PDF into LangChain document objects
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split the PDF text into manageable chunks
    # Large documents need splitting to fit into token limits of embeddings/LLMs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Convert text chunks into embeddings using OpenAI’s embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Option: use "text-embedding-3-large" for higher accuracy

    # Store embeddings inside a Chroma vector database (local persistent DB)
    vectordb = Chroma.from_documents(
        docs,
        embeddings
        #persist_directory="./langchain_db"  # saves to disk for reuse
    )
    return vectordb

# --- Load or create the vectorstore ---
vectordb = load_vectorstore()

# --- Build the Retrieval-QA pipeline ---
retriever = vectordb.as_retriever()   # Converts vector DB into retriever interface
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# gpt-4o-mini is used for fast + cheap Q&A, temperature=0 = deterministic answers

# Create a RetrievalQA chain that:
# 1. Uses retriever to fetch relevant chunks from PDF
# 2. Passes them to the LLM to generate a final answer
# 3. Returns both the answer and the source documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit input and output ---
# A text input box for the user to type a question
question = st.text_input("Ask a question about the Dash:")

# When the user asks something, run the RetrievalQA pipeline
if question:
    result = qa_chain({"query": question})

    # Display the LLM’s final answer
    st.write("### Answer")
    st.write(result["result"])

    # Show the source passages retrieved from the PDF
    with st.expander("Sources"):
        for i, doc in enumerate(result["source_documents"], start=1):
            st.write(f"**Source {i}:**")
            # Truncate to first 300 characters for readability
            st.write(doc.page_content[:300] + "...")
