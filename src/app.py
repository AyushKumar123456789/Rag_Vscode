import os
import torch
from sentence_transformers import SentenceTransformer
from download_pdf import download_pdf
from process_pdf import process_pdf
from generate_embeddings import generate_embeddings
from retrieve import load_embeddings, retrieve_relevant_chunks, print_results

def main():
    # Paths
    pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    data_dir = "../data"
    pdf_path = os.path.join(data_dir, "human-nutrition-text.pdf")
    chunks_csv_path = os.path.join(data_dir, "text_chunks.csv")
    embeddings_csv_path = os.path.join(data_dir, "text_chunks_without_embeddings.csv")
    embeddings_npy_path = os.path.join(data_dir, "embeddings.npy")

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Download PDF
    download_pdf(pdf_url, pdf_path)

    # Process PDF to extract text chunks
    if os.path.exists(chunks_csv_path):
        print(f"Chunks file already exist at {chunks_csv_path} , So no need to create it again.")
    else:
        print("Processing PDF to extract text chunks...")
        process_pdf(pdf_path, chunks_csv_path)

    # Generate embeddings
    if os.path.exists(embeddings_npy_path) and os.path.exists(embeddings_csv_path) :
        print(f"Embeddings file already exist at {embeddings_npy_path} , So no need to create it again.")
    else :
        print("Generating embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generate_embeddings(chunks_csv_path, embeddings_csv_path, embeddings_npy_path, device)

    # Load embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, embeddings = load_embeddings(embeddings_csv_path, embeddings_npy_path, device)
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # Main loop
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() in ['exit', 'quit']:
            break
        indices, scores = retrieve_relevant_chunks(query, df, embeddings, model)
        print_results(query, df, indices, scores)

if __name__ == "__main__":
    main()
