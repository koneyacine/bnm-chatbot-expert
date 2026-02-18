import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import json

# Configuration
EXCEL_PATH = os.getenv("EXCEL_PATH", "data/DocsBnmQR.xlsx")
JSONL_PATH = os.getenv("JSONL_PATH", "bnm_dataset.jsonl")
DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bnm_qa")

def ingest():
    documents = []
    metadatas = []
    ids = []

    print(f"Trying to read Excel file: {EXCEL_PATH}")
    try:
        # Load the Excel file
        df = pd.read_excel(EXCEL_PATH)
        
        question_col = None
        answer_col = None
        
        for col in df.columns:
            clean_col = str(col).replace('\x00', '').strip().lower()
            if 'question' in clean_col:
                question_col = col
            elif 'ponse' in clean_col:
                answer_col = col
        
        if question_col and answer_col:
            print(f"Detected columns: Question -> '{question_col}', Réponse -> '{answer_col}'")
            for i, row in df.iterrows():
                q = str(row[question_col])
                a = str(row[answer_col])
                documents.append(f"Question: {q}\nRéponse: {a}")
                metadatas.append({"question": q, "answer": a})
                ids.append(f"id_{i}")
        else:
            print("Could not identify columns in Excel.")

    except Exception as e:
        print(f"Excel access failed: {e}")
        if os.path.exists(JSONL_PATH):
            print(f"Falling back to JSONL file: {JSONL_PATH}")
            try:
                with open(JSONL_PATH, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        if len(messages) >= 3:
                            q = messages[1]["content"]
                            a = messages[2]["content"]
                            documents.append(f"Question: {q}\nRéponse: {a}")
                            metadatas.append({"question": q, "answer": a})
                            ids.append(f"id_jsonl_{i}")
            except Exception as e2:
                print(f"JSONL access also failed: {e2}")

    if not documents:
        print("No data found to ingest.")
        return

    try:
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        # Upsert into ChromaDB
        print(f"Ingesting {len(documents)} Q&A pairs (Combined format)...")
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
