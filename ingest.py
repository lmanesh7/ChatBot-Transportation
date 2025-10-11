# file: ingest.py (Upgraded Version)

import os
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader

load_dotenv()

MONGO_URI = os.environ["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client.transportationDB
collection = db.knowledge_base
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def ingest_data(loader):
    """Takes a LangChain loader, splits the data, creates embeddings, and stores them."""
    print(f"Loading data with {type(loader).__name__}...")
    documents = loader.load()
    
    # Use a more robust text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        print("No documents were extracted. Aborting.")
        return

    # Extract text content for embedding
    texts = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(texts)
    
    to_insert = []
    for i, doc in enumerate(docs):
        doc.metadata['text'] = doc.page_content # Keep original text for review
        to_insert.append({"text": doc.page_content, "embedding": doc_embeddings[i], "metadata": doc.metadata})
    
    collection.insert_many(to_insert)
    print(f"Successfully ingested {len(docs)} document chunks.")

if __name__ == "__main__":
    # --- CHOOSE YOUR DATA SOURCE ---
    
    # Option 1: Ingest from a local text file
    # print("Ingesting from knowledge.txt...")
    # text_loader = TextLoader("knowledge.txt")
    # ingest_data(text_loader)
    
    # Option 2: Ingest from a PDF file (place a DOT manual in your folder)
    # print("Ingesting from MODOT_Manual.pdf...")
    # pdf_loader = PyPDFLoader("MODOT_Manual.pdf")
    # ingest_data(pdf_loader)

    # Option 3: Ingest from a web page
    print("Ingesting from MoDOT news page...")
    web_loader = WebBaseLoader("https://www.modot.org/news-and-media")
    ingest_data(web_loader)