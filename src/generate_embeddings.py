from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
import os

def generate_embeddings(csv_path: str, embeddings_csv_path: str, embeddings_npy_path: str, device: str = "cpu"):
    """Generates embeddings for text chunks and saves them."""
    df = pd.read_csv(csv_path)
    model = SentenceTransformer("all-mpnet-base-v2", device=device) #Initializes a pre-trained model ("all-mpnet-base-v2") from sentence_transformers.
    text_chunks = df["sentence_chunk"].tolist()
    embeddings = model.encode(text_chunks, batch_size=32, convert_to_tensor=True)

    # Save embeddings as .npy file
    np.save(embeddings_npy_path, embeddings.cpu().numpy())
    print(f"Embeddings saved to {embeddings_npy_path}")

    # Save dataframe without embeddings to CSV
    df.to_csv(embeddings_csv_path, index=False)
    print(f"Dataframe saved to {embeddings_csv_path}")
