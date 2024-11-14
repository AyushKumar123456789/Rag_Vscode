from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
import textwrap
import requests
import os
import google.generativeai as genai
import random
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
import torch.nn as nn

genai.configure(api_key="AIzaSyDSSsA-hrjCFfKVkwrI50tnljy0WCIRpsU")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)
history=[]
chat_session = model.start_chat(
  history=history
)

def load_embeddings(embeddings_csv_path: str, embeddings_npy_path: str, device: str = "cpu"):
    """Loads embeddings and text chunks from files."""
    df = pd.read_csv(embeddings_csv_path)
    embeddings = np.load(embeddings_npy_path)
    print(f"Embeddings loaded with shape: {embeddings.shape}")
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    print(f"Embeddings tensor shape: {embeddings.shape}")
    return df, embeddings


def get_synonyms(query: str):
    """Returns synonyms for the input query using WordNet."""
    synonyms = set()
    for word in query.split():
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)


def expand_query_with_hde(query: str):
    """Generates expanded queries using synonyms (Hypothetical Document Embeddings)."""
    synonyms = get_synonyms(query)
    if not synonyms:
        return [query]  # If no synonyms, return original query
    expanded_queries = set()
    # Expand the query by adding a random synonym for each word in the query
    for _ in range(3):  # Generate 3 variations of the query
        new_query = []
        for word in query.split():
            # Randomly pick a synonym for the word or use the original word
            if random.random() < 0.5 and synonyms:  # 50% chance to expand
                new_query.append(random.choice(synonyms))
            else:
                new_query.append(word)
        expanded_queries.add(" ".join(new_query))
    return list(expanded_queries)


def retrieve_relevant_chunks_with_expansion(query: str, df: pd.DataFrame, embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 3):
    """Retrieve relevant chunks for the original query and its expanded versions."""
    # Generate expanded queries using Hypothetical Document Embeddings (HDE)
    expanded_queries = expand_query_with_hde(query)
    print(f"Expanded Queries: {expanded_queries}")
    
    all_indices = []
    all_scores = []
    
    for expanded_query in expanded_queries:
        query_embedding = model.encode(expanded_query, convert_to_tensor=True)
        projection_layer = nn.Linear(768, 384)
        projected_embeddings = projection_layer(embeddings)
        query_embedding = query_embedding.to("cpu")
        projected_embeddings = projected_embeddings.to("cpu")
        dot_scores = util.dot_score(query_embedding, projected_embeddings)[0]
        top_results = torch.topk(dot_scores, k=top_k)

        # Collect the results from each expanded query
        all_indices.append(top_results.indices.detach().cpu().numpy())
        all_scores.append(top_results.values.detach().cpu().numpy())
    
    # Flatten the results (you can apply more sophisticated methods to combine results)
    indices = np.concatenate(all_indices)
    scores = np.concatenate(all_scores)
    return indices, scores


def retrieve_relevant_chunks(query: str, df: pd.DataFrame, embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 3):
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


def query_gemini_with_retrieved_chunks(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, original_query: str):
    """Uses retrieved chunks as context for the Gemini API and prints the responses."""
    # Combine the original query with relevant chunks: query expansion technique
    combined_query = "Based on the context chunk information and ignore media elemnts, answer the query: '{original_query}'. Ignore any media or non-relevant information, and focus on the textual content. Make sure to provide a clear and concise answer based on the relevant information."
    for idx, score in zip(indices, scores):
        text = df.iloc[idx]["sentence_chunk"]
        combined_query += f"\nScore: {score:.4f}\nText: {text}\n\n"
    response = model.generate_content(combined_query)
    if response.text:
        response_data = response.text
        print("Combined Query to Gemini:\n")
        print(combined_query)
        print("Gemini Response:\n")
        print(response_data)
        print("\n" + "=" * 150 + "\n")
        print(f"Mean Score: {np.mean(scores):.4f}") 
    else:
        print(f"Error: {response.status_code}, {response.text}")


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)