import os
import shutil
import openai
import faiss
import psycopg2
import numpy as np
from PyPDF2 import PdfReader
from pathlib import Path

# Configure your OpenAI API key
openai.api_key = "your_openai_api_key"

# Define folders
INPUT_FOLDER = "./pdf_files"
PROCESSED_FOLDER = "./processed_files"

# FAISS setup
vector_dimension = 1536  # Adjust this if using a different OpenAI embedding model
index = faiss.IndexFlatL2(vector_dimension)

# PostgreSQL database setup
db_params = {
    'dbname': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'port': 'your_port'
}
conn = psycopg2.connect(**db_params)
c = conn.cursor()

# Create tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS processed_files (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE,
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )''')
conn.commit()

# Ensure processed folder exists
Path(PROCESSED_FOLDER).mkdir(exist_ok=True)

# Function to extract text from PDF
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = "\n".join([page.extract_text() for page in reader.pages])
    return text

# Function to get vector from text using OpenAI
def get_text_vector(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"]).astype('float32')

# Process new PDF files
def process_pdf_files():
    input_path = Path(INPUT_FOLDER)
    processed_path = Path(PROCESSED_FOLDER)

    for pdf_file in input_path.glob("*.pdf"):
        filename = pdf_file.name

        # Skip if the file has already been processed
        c.execute("SELECT id FROM processed_files WHERE filename = %s", (filename,))
        if c.fetchone():
            continue

        try:
            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_file)

            # Convert text to vector
            vector = get_text_vector(text)

            # Add vector to FAISS index
            index.add(np.array([vector]))

            # Log success in the database
            c.execute("INSERT INTO processed_files (filename, status) VALUES (%s, %s)", (filename, "success"))
            conn.commit()

            # Move the file to the processed folder
            shutil.move(str(pdf_file), processed_path / filename)
            print(f"Processed and moved: {filename}")
        except Exception as e:
            # Log failure in the database
            c.execute("INSERT INTO processed_files (filename, status, error_message) VALUES (%s, %s, %s)", (filename, "failed", str(e)))
            conn.commit()
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    while True:
        process_pdf_files()
        print("Waiting for new files...")
        time.sleep(10)
