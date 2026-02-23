import chromadb
from chromadb.utils import embedding_functions
import os

DB_PATH = "chroma_db"
COLLECTION_NAME = "bnm_qa"

def verify():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    
    count = collection.count()
    print(f"Total documents in collection: {count}")
    
    # Get all documents with source metadata
    results = collection.get()
    
    sources = set()
    for meta in results['metadatas']:
        sources.add(meta.get('source'))
    
    print(f"Sources found: {sources}")
    
    if "manual_qa.json" in sources:
        print("SUCCESS: manual_qa.json is present in the database.")
    else:
        print("FAILURE: manual_qa.json is NOT found in the database.")

    # Sample query for "Qui es-tu ?"
    query = "Qui es-tu ?"
    print(f"\nTesting query: '{query}'")
    query_results = collection.query(query_texts=[query], n_results=1)
    
    if query_results['documents'] and len(query_results['documents'][0]) > 0:
        print(f"Top result: {query_results['documents'][0][0]}")
        print(f"Source: {query_results['metadatas'][0][0].get('source')}")
        print(f"Distance: {query_results['distances'][0][0]}")

if __name__ == "__main__":
    verify()
