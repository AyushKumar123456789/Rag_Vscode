# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Python. It allows users to query a PDF textbook on Human Nutrition and retrieves relevant information.

## Features

- Extracts text from a PDF and processes it into chunks.
- Generates embeddings for text chunks using Sentence Transformers.
- Retrieves relevant text chunks based on user queries.
- Optionally displays the corresponding PDF pages.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag_chatbot.git
cd rag_chatbot
```
## Steps to start the app
python3 -m venv venv  
source venv/bin/activate  (MacOS)
pip3 install -r requirements.txt
cd src
python3 app.py



# additional install
pip3 install 'accelerate>=0.26.0'
pip3 uninstall sentence-transformers transformers 
pip3 install sentence-transformers==3.0.1  transformers==4.44
