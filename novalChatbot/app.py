import streamlit as st
import os
import pickle
from chatbot import (
    extract_text_from_pdf,  # Extract/split PDF (with chunking & overlap)
    get_completion,  # Call OpenAI Chat Completions
    build_prompt,  # Build prompt template
    identify_question_type,  # Identify question type (affects retrieval depth)
    RagBot,  # RAG pipeline wrapper (retrieval + prompt + LLM)
    MyVectorDBConnector,  # Vector DB wrapper (Chroma persistent)
    OpenAIEmbeddingFunction  # Embedding wrapper
)

DEFAULT_PDF_PATH = "westJourney.pdf"


def initialize_bot():
    """Initialize the RAG bot with either preprocessed data or by processing the default PDF in real-time."""
    if 'bot' not in st.session_state:
        try:
            processed_data_path = "processed_data.pkl"
            chroma_dir = "./chroma_db"

            if os.path.exists(processed_data_path) and os.path.exists(chroma_dir):
                # Load preprocessed vector database metadata
                with open(processed_data_path, 'rb') as f:
                    data = pickle.load(f)

                vector_db = MyVectorDBConnector(
                    data['collection_name'],
                    embedding_function=OpenAIEmbeddingFunction(),
                    persist_directory=chroma_dir
                )
                st.success("Preprocessed data loaded successfully")
            else:
                # Fallback: process PDF in real time
                st.warning("No preprocessed data found, processing PDF in real-time...")
                if not os.path.exists(DEFAULT_PDF_PATH):
                    st.error(f"Default file {DEFAULT_PDF_PATH} not found")
                    return False

                vector_db = MyVectorDBConnector(
                    "journey",
                    embedding_function=OpenAIEmbeddingFunction(),
                    persist_directory=chroma_dir
                )

                with st.spinner("Processing PDF file..."):
                    paragraphs = extract_text_from_pdf(DEFAULT_PDF_PATH, min_line_length=10)
                    if not paragraphs:
                        st.error("Could not extract text from PDF")
                        return False
                    vector_db.add_documents(paragraphs)

            # Create the bot instance
            st.session_state.bot = RagBot(
                vector_db,
                llm_api=get_completion,
                n_results=12
            )
            st.success("Document loaded successfully")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return False
    return True


def process_uploaded_file(uploaded_file):
    """Handle a user-uploaded PDF: save, process, and reinitialize the bot."""
    try:
        # Save uploaded file temporarily
        with open("dashQA.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("File uploaded successfully!")

        vector_db = MyVectorDBConnector(
            "new_book",
            embedding_function=OpenAIEmbeddingFunction(),
            persist_directory="./chroma_db"
        )

        # Extract and add text to the vector DB
        with st.spinner("Processing uploaded file..."):
            paragraphs = extract_text_from_pdf("dashQA.pdf", min_line_length=10)
            if not paragraphs:
                st.error("Could not extract text from uploaded PDF")
                return False
            vector_db.add_documents(paragraphs)

        # Replace existing bot with a new instance bound to this document
        st.session_state.bot = RagBot(
            vector_db,
            llm_api=get_completion,
            n_results=12
        )
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False
    finally:
        # Always clean up the temp file
        if os.path.exists("dashQA.pdf"):
            os.remove("dashQA.pdf")


def display_chat_history():
    """Render chat history from the Streamlit session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(prompt):
    """Handle user input: append to history, query the bot, and display the answer."""
    if not initialize_bot():
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.chat(prompt)
            st.markdown(response)

    # Save assistant response in history
    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="Novel Q&A Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("Novel Q&A Assistant 📚")

    # Instructions panel
    with st.expander("Instructions", expanded=True):
        st.markdown("""
        ### Features
        1. Default novel text is pre-loaded
        2. You can ask about:
           - Characters ("What is the personality of the main character?")
           - Plot ("What happened in Chapter 1?")
           - Setting ("Where does the story take place?")
           - Themes ("What are the main themes of the story?")
        3. You can also upload your own PDF for analysis

        ### Tips
        - Be specific with your questions
        - Uploaded PDFs must be text-searchable
        - All answers are based on the document content
        """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Document Settings")
        uploaded_file = st.file_uploader("Upload another PDF (optional)", type=['pdf'])
        if uploaded_file:
            if process_uploaded_file(uploaded_file):
                st.success("Document switched successfully!")

    # Initialize chat history if not set
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    display_chat_history()

    # Chat input box
    if prompt := st.chat_input("Enter your question"):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
