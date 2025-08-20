import pickle
import shutil
import os
from tqdm import tqdm
from chatbot import MyVectorDBConnector, OpenAIEmbeddingFunction, extract_text_from_pdf

def preprocess_pdf(
        pdf_path="westJourney.pdf",
        output_path="processed_data.pkl",
        chroma_dir="./chroma_db"
):
    """Preprocess PDF and save results"""
    print("Starting PDF preprocessing...")

    # Clean old data
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    # Create vector database
    vector_db = MyVectorDBConnector(
        "journey",
        embedding_function=OpenAIEmbeddingFunction(),
        persist_directory=chroma_dir
    )

    # Extract text
    print("Extracting text from PDF...")
    paragraphs = extract_text_from_pdf(pdf_path, min_line_length=10)

    # ✅ Filter out empty, None, or non-string paragraphs
    paragraphs = [p for p in paragraphs if p and isinstance(p, str) and p.strip()]

    if not paragraphs:
        print("No valid text extracted from PDF")
        return False

    print(f"Extracted {len(paragraphs)} paragraphs")

    # Batch processing
    batch_size = 50  # Process 50 paragraphs per batch
    total_batches = (len(paragraphs) + batch_size - 1) // batch_size

    print("Building vector database...")
    for i in tqdm(range(0, len(paragraphs), batch_size), total=total_batches, desc="Processing"):
        batch = paragraphs[i:i + batch_size]

        # ✅ Skip empty batch
        if not batch:
            print(f"Skipping empty batch {i // batch_size + 1}/{total_batches}")
            continue

        # ✅ Debug info
        print(f"\n===> Processing batch {i // batch_size + 1}/{total_batches}")
        print(f"Paragraph count: {len(batch)}")
        print(f"First paragraph (100 chars): {batch[0][:100]}...")
        print(f"Paragraph type: {type(batch[0])}")

        try:
            vector_db.add_documents(batch, ids=[str(j) for j in range(i, i + len(batch))])
        except Exception as e:
            print(f"Error in batch {i // batch_size + 1}/{total_batches}: {str(e)}")
            continue

    # Save processed results
    print("Saving processed results...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'paragraphs': paragraphs,
            'collection_name': 'journey'
        }, f)

    print("Processing completed!")
    print(f"- Vector database saved at: {chroma_dir}")
    print(f"- Text data saved at: {output_path}")
    return True

if __name__ == "__main__":
    # Add command line argument support
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess a PDF file')
    parser.add_argument('--pdf', type=str, default="westJourney.pdf", help='Path to PDF file')
    parser.add_argument('--output', type=str, default="processed_data.pkl", help='Output file path')
    parser.add_argument('--chroma', type=str, default="./chroma_db", help='Vector database directory')

    args = parser.parse_args()

    # Run preprocessing
    success = preprocess_pdf(
        pdf_path=args.pdf,
        output_path=args.output,
        chroma_dir=args.chroma
    )

    if success:
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing failed!")
