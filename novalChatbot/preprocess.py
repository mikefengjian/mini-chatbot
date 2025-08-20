import pickle
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def extract_text_from_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    return text_splitter.split_documents(docs)


def main():
    pdf_path = "westJourney.pdf"
    output_dir = "faiss_index"
    metadata_path = "processed_data.pkl"

    print("Starting PDF preprocessing...")
    paragraphs = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(paragraphs)} paragraphs")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(paragraphs, embeddings)

    # 保存向量数据库
    vectorstore.save_local("faiss_index")

    # 保存元数据
    with open(metadata_path, "wb") as f:
        pickle.dump(paragraphs, f)

    print("Processing completed!")
    print(f"- FAISS index saved at: {output_dir}")
    print(f"- Metadata saved at: {metadata_path}")


if __name__ == "__main__":
    main()
