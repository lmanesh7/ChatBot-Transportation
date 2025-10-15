# file: ingest.py (Corrected and Upgraded)

import os
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated LangChain imports for v0.2+
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader

load_dotenv()

# Explicitly load the Google API Key for clarity
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- Error Handling for Environment Variables ---
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables. Please check your .env file.")

client = MongoClient(MONGO_URI)
db = client.transportationDB
collection = db.knowledge_base
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# file: ingest.py (Version with Enhanced Debugging)

# (Keep all your imports and initial setup the same)
# ...

def ingest_data_in_batches(loader, batch_size=100):
    """
    Takes a LangChain loader, splits data, creates embeddings in batches,
    and stores them in MongoDB. NOW WITH ENHANCED DEBUGGING!
    """
    print("--- Starting Ingestion Process ---")
    
    # STEP 1: Load the data
    print(f"1. Loading data with {type(loader).__name__}...")
    try:
        documents = loader.load()
        if not documents:
            print("❌ FAILURE: Loader returned no documents. Please check the file path and ensure the PDF contains selectable text.")
            return
        print(f"✅ SUCCESS: Loaded {len(documents)} document(s) from the source.")
    except Exception as e:
        print(f"❌ FAILURE: An error occurred during the loading step: {e}")
        return

    # STEP 2: Split the documents into chunks
    print("\n2. Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        print("❌ FAILURE: Text splitter created 0 chunks. The source document might not contain enough extractable text.")
        return
    
    total_chunks = len(docs)
    print(f"✅ SUCCESS: Created a total of {total_chunks} document chunks.")

    # STEP 3: Process chunks in batches
    print(f"\n3. Processing {total_chunks} chunks in batches of {batch_size}...")
    for i in range(0, total_chunks, batch_size):
        current_batch_num = (i // batch_size) + 1
        print(f"\n--- Processing Batch {current_batch_num} ---")
        
        batch_docs = docs[i:i + batch_size]
        print(f"   - This batch contains {len(batch_docs)} chunks.")
        
        texts = [doc.page_content for doc in batch_docs]

        try:
            print("   - Creating embeddings for the batch...")
            doc_embeddings = embeddings.embed_documents(texts)
            print("   ✅ Embeddings created successfully.")
            
            to_insert = []
            for j, doc in enumerate(batch_docs):
                to_insert.append({
                    "text": doc.page_content, 
                    "embedding": doc_embeddings[j], 
                    "metadata": doc.metadata
                })
            
            if to_insert:
                print(f"   - Inserting {len(to_insert)} records into MongoDB...")
                result = collection.insert_many(to_insert)
                print(f"   ✅ SUCCESS: Inserted {len(result.inserted_ids)} records into the database.")
            else:
                print("   ⚠️ WARNING: No documents were prepared for insertion in this batch.")

        except Exception as e:
            print(f"   ❌ FAILURE: An error occurred while processing Batch {current_batch_num}: {e}")
            continue 
            
    print("\n--- Ingestion Script Finished ---")

# Remember to call the new function in your main block
if __name__ == "__main__":
    print("Ingesting from QnA.pdf...")
    pdf_loader = PyPDFLoader("QnA.pdf")
    ingest_data_in_batches(pdf_loader)