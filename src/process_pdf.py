import fitz  # PyMuPDF
from tqdm.auto import tqdm
from spacy.lang.en import English
import pandas as pd
import re
import os

def text_formatter(text: str) -> str:
    """Performs minor formatting on text.Cleans the text by replacing newlines with spaces and stripping leading/trailing whitespace."""
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> list:
    """Opening the PDF file and extracting text from each page. """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        """Iterates over each page and extracts the text."""
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number - 41,
            "text": text
        })
    return pages_and_texts

def split_sentences(pages_and_texts: list, num_sentence_chunk_size: int = 10) -> list:
    """Splits text into sentences and groups them into chunks."""
    nlp = English() #initalising the spacy English model , spacy is used to create sentences chunk.
    nlp.add_pipe("sentencizer") # Adding the sentencizer to the pipeline.
    for item in tqdm(pages_and_texts):
        doc = nlp(item["text"])
        sentences = [str(sent) for sent in doc.sents] #converting to string extra check
        item["sentence_chunks"] = [
            sentences[i:i + num_sentence_chunk_size] 
            for i in range(0, len(sentences), num_sentence_chunk_size)
        ]
    return pages_and_texts

def prepare_chunks(pages_and_texts: list) -> list:
    """Join each chunk into single string and remove spaces between them and after . give space using regex"""
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def save_chunks_to_csv(pages_and_chunks: list, csv_path: str):
    """Converts the list of chunks into a pandas DataFrame and saves it as a CSV file."""
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(csv_path, index=False)
    print(f"Chunks saved to {csv_path}")

def process_pdf(pdf_path: str, csv_path: str):
    """Full processing pipeline.The main function that calls all the other functions in sequence to process the PDF."""
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_texts = split_sentences(pages_and_texts)
    pages_and_chunks = prepare_chunks(pages_and_texts)
    save_chunks_to_csv(pages_and_chunks, csv_path)

"""
Understanding the Flow:

1. Extract Text: The PDF is opened, and text is extracted from each page.
2. Split into Sentences: The extracted text is split into sentences using spaCy.
3. Group Sentences: Sentences are grouped into chunks (e.g., every 10 sentences form a chunk).
4. Clean and Prepare Chunks: Chunks are cleaned up to ensure proper formatting.
5. Save to CSV: The chunks are saved to a CSV file for later use.
"""