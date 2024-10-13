import os
import requests

def download_pdf(pdf_url: str, pdf_path: str):
    """Downloads the PDF file if it doesn't exist."""
    if not os.path.exists(pdf_path):
        print("File doesn't exist, downloading...")
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)
            print(f"The file has been downloaded and saved as {pdf_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"File {pdf_path} exists.")
