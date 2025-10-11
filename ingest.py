# file: ingest.py (Updated for Google)
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# --- Ensure you have your GOOGLE_API_KEY in the .env file ---
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

MONGO_URI = os.environ["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client.transportationDB
collection = db.knowledge_base

# Clear the collection before ingesting new data
collection.delete_many({})
print("Cleared existing documents in knowledge_base.")

with open("knowledge.txt") as f:
    knowledge_text = f.read()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_text(knowledge_text)

# Use Google's embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
doc_embeddings = embeddings.embed_documents(docs)

to_insert = [{"text": doc, "embedding": embedding} for doc, embedding in zip(docs, doc_embeddings)]

collection.insert_many(to_insert)
print(f"Successfully ingested {len(docs)} documents using Google's embedding model.")