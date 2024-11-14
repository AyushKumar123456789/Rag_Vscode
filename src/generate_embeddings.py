from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(csv_path: str, embeddings_csv_path: str, embeddings_npy_path: str, device: str = "cpu"):
    """Generates embeddings for text chunks and saves them, comparing original and fine-tuned models."""
    df = pd.read_csv(csv_path)

    # Load the fine-tuned model
    # fine_tuned_model_path = "fine_tuned_embedding_model/model"
    # fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)

    # Load the original model
    original_model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # List of sentence chunks
    text_chunks = df["sentence_chunk"].tolist()

    # Generate embeddings from both models
    # fine_tuned_embeddings = fine_tuned_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)
    original_embeddings = original_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)

    # Convert embeddings to numpy arrays for cosine similarity calculation
    # fine_tuned_embeddings_np = fine_tuned_embeddings.cpu().numpy()
    original_embeddings_np = original_embeddings.cpu().numpy()

    # Compute cosine similarity between the fine-tuned model and the original model embeddings
    # cosine_similarities = cosine_similarity(fine_tuned_embeddings_np, original_embeddings_np)

    # Calculate the average cosine similarity score for the entire batch
    # avg_cosine_similarity = np.mean(cosine_similarities)
    # print(f"Average Cosine Similarity Score between Fine-Tuned and Original Model: {avg_cosine_similarity}")

    # Save the embeddings from the original embedding model
    np.save(embeddings_npy_path, original_embeddings.cpu().numpy())
    print(f"Embeddings saved to {embeddings_npy_path}")

    # Save the embeddings from the fine-tuned embedding model
    # np.save(embeddings_npy_path, fine_tuned_embeddings.cpu().numpy())
    # print(f"Embeddings saved to {embeddings_npy_path}")

    # Save the dataframe to CSV
    df.to_csv(embeddings_csv_path, index=False)
    print(f"Dataframe saved to {embeddings_csv_path}")
