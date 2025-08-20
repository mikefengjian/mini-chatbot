import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from chatbot import (
    extract_text_from_pdf,
    get_completion,
    build_prompt,
    identify_question_type,
    RagBot,
)

DEFAULT_PDF_PATH = os.path.join("novalChatbot", "westJourney.pdf")

def initialize_bot():
    if "bot" not in st.session_state:
        try:
            processed_data_path = "processed_data.pkl"
            faiss_dir = "faiss_index"

            if os.path.exists(processed_data_path) and os.path.exists(faiss_dir):
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_db = FAISS.load_local(
                    faiss_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success("Preprocessed data loaded successfully")
            else:
                st.warning("No preprocessed data found, processing PDF in real-time...")
                if not os.path.exists(DEFAULT_PDF_PATH):
                    st.error(f"Default file {DEFAULT_PDF_PATH} not found")
                    return False

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                with st.spinner("Processing PDF file..."):
                    paragraphs = extract_text_from_pdf(DEFAULT_PDF_PATH, min_line_length=10)
                    if not paragraphs:
                        st.error("Could not extract text from PDF")
                        return False
                    vector_db = FAISS.from_texts(paragraphs, embeddings)
                    vector_db.save_local(faiss_dir)

            st.session_state.bot = RagBot(
                vector_db,
                llm_api=get_completion,
                n_results=12,
            )
            st.success("Document loaded successfully")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return False
    return True


def process_uploaded_file(uploaded_file):
    try:
        temp_path = "uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("File uploaded successfully!")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        with st.spinner("Processing uploaded file..."):
            paragraphs = extract_text_from_pdf(temp_path, min_line_length=10)
            if not paragraphs:
                st.error("Could not extract text from uploaded PDF")
                return False
            vector_db = FAISS.from_texts(paragraphs, embeddings)
            vector_db.save_local("faiss_index")

        st.session_state.bot = RagBot(
            vector_db,
            llm_api=get_completion,
            n_results=12,
        )
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False
    finally:
        if os.path.exists("uploaded.pdf"):
            os.remove("uploaded.pdf")


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(prompt):
    if not initialize_bot():
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.chat(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="Novel Q&A Assistant", page_icon="📚", layout="wide")
    st.title("Novel Q&A Assistant 📚")

    with st.expander("Instructions", expanded=True):
        st.markdown(""" 
        ### Features 
         1. Default novel text is pre-loaded
         2. You can ask about: 
          - Characters ("What is the personality of the main character?") 
          - Plot ("What happened in Chapter 1?")
          - Setting ("Where does the story take place?") 
          - Themes ("What are the main themes of the story?") 
         3. You can also upload your own PDF for analysis ### 
         
         Tips
          - Be specific with your questions 
          - Uploaded PDFs must be text-searchable 
          - All answers are based on the document content """)

    with st.sidebar:
        st.header("Document Settings")
        uploaded_file = st.file_uploader("Upload another PDF (optional)", type=["pdf"])
        if uploaded_file:
            if process_uploaded_file(uploaded_file):
                st.success("Document switched successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat_history()

    if prompt := st.chat_input("Enter your question"):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
