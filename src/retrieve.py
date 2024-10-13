from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
import textwrap

def load_embeddings(embeddings_csv_path: str, embeddings_npy_path: str, device: str = "cpu"):
    """Loads embeddings and text chunks from files."""
    df = pd.read_csv(embeddings_csv_path)
    embeddings = np.load(embeddings_npy_path)
    print(f"Embeddings loaded with shape: {embeddings.shape}")
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    print(f"Embeddings tensor shape: {embeddings.shape}")
    return df, embeddings


def retrieve_relevant_chunks(query: str, df: pd.DataFrame, embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 5):
    """Retrieves the most relevant text chunks for a given query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    top_results = torch.topk(dot_scores, k=top_k)
    return top_results.indices.cpu().numpy(), top_results.values.cpu().numpy()


def print_results(query: str, df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray):
    """Prints the retrieved results."""
    print(f"Query: '{query}'\n")
    print("Results:")
    for idx, score in zip(indices, scores):
        text = df.iloc[idx]["sentence_chunk"]
        page_number = df.iloc[idx]["page_number"]
        print(f"Score: {score:.4f}")
        print_wrapped(f"Text: {text}")
        print(f"Page number: {page_number}\n")

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)
