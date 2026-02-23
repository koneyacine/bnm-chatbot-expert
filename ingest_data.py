import chromadb
from chromadb.utils import embedding_functions
import os
import json
from document_processor import process_documents

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bnm_qa")

def ingest():
    print(f"Scanning directory: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return

    # Process documents into chunks
    processed_docs = process_documents(DATA_DIR)
    
    documents = [doc["content"] for doc in processed_docs]
    metadatas = [doc["metadata"] for doc in processed_docs]
    ids = [doc["id"] for doc in processed_docs]

    # Process manual QA from JSON if exists
    JSON_PATH = "manual_qa.json"
    if os.path.exists(JSON_PATH):
        print(f"Loading manual QA from: {JSON_PATH}")
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                manual_qa = json.load(f)
                for i, qa in enumerate(manual_qa):
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    if q and a:
                        content = f"Question: {q}\n\nRéponse: {a}"
                        documents.append(content)
                        metadatas.append({"source": "manual_qa.json", "type": "qa", "qa_id": i})
                        ids.append(f"manual_qa_{i}")
            print(f"Added {len(manual_qa)} QA pairs from JSON.")
        except Exception as e:
            print(f"Error loading {JSON_PATH}: {e}")

    if not documents:
        print("No documents found to ingest.")
        return

    try:
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Reset or update collection
        # Note: If you want to start fresh, you can use client.delete_collection(COLLECTION_NAME) beforehand
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        # Upsert into ChromaDB
        print(f"Ingesting {len(documents)} chunks into ChromaDB...")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion complete!")

    except Exception as e:
        print(f"An error occurred during DB operation: {e}")

if __name__ == "__main__":
    ingest()
