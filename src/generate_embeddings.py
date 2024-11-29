from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings_original_model(csv_path: str, embeddings_csv_path: str, embeddings_npy_path: str, device: str = "cpu"):
    """Generates embeddings for text chunks using the original model and saves them."""
    df = pd.read_csv(csv_path)
    original_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    text_chunks = df["sentence_chunk"].tolist()
    original_embeddings = original_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)
    original_embeddings_np = original_embeddings.cpu().numpy()
    np.save(embeddings_npy_path, original_embeddings.cpu().numpy())
    print(f"Original model embeddings saved to {embeddings_npy_path}")
    df.to_csv(embeddings_csv_path, index=False)
    print(f"Dataframe saved to {embeddings_csv_path}")


def generate_embeddings_fine_tuned_model(csv_path: str, embeddings_csv_path: str, embeddings_npy_path: str, fine_tuned_model_path: str, device: str = "cpu"):
    """Generates embeddings for text chunks using the fine-tuned model and saves them."""
    df = pd.read_csv(csv_path)
    fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)
    text_chunks = df["sentence_chunk"].tolist()
    fine_tuned_embeddings = fine_tuned_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)
    fine_tuned_embeddings_np = fine_tuned_embeddings.cpu().numpy()
    np.save(embeddings_npy_path, fine_tuned_embeddings.cpu().numpy())
    print(f"Fine-tuned model embeddings saved to {embeddings_npy_path}")
    df.to_csv(embeddings_csv_path, index=False)
    print(f"Dataframe saved to {embeddings_csv_path}")
