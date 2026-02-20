import os
import json
import ollama
from document_processor import process_documents

# Configuration
DATA_DIR = "data"
MANUAL_QA_FILE = "manual_qa.json"
OUTPUT_FILE = "bnm_dataset.jsonl"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen2.5:7b" 

client = ollama.Client(host=OLLAMA_HOST)

def generate_qa_pair(chunk_content):
    """Génère une question et une réponse avec un prompt renforcé."""
    prompt = f"""Tu es un expert en données pour la Banque Nationale de Mauritanie (BNM).
À partir du texte suivant, génère une paire Question/Réponse pertinente.

DIRECTIVES :
1. Basé UNIQUEMENT sur le texte fourni, pas d'hallucination.
2. La question doit ressembler à celle d'un client BNM (en français ou en arabe si approprié).
3. La réponse doit être précise, professionnelle et suivre l'identité BNM.
4. Si le texte contient une table, pose une question sur une donnée spécifique de la table.

TEXTE :
{chunk_content}

Formate ta réponse en JSON valide : {{"question": "...", "answer": "..."}}
JSON :"""
    
    try:
        response = client.generate(model=MODEL, prompt=prompt)
        raw_text = response['response']
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(raw_text[start:end])
    except Exception as e:
        print(f"Erreur QA: {e}")
    return None

def prepare_dataset():
    dataset = []

    # 1. Charger les QA manuelles si elles existent
    if os.path.exists(MANUAL_QA_FILE):
        print(f"Chargement des QA manuelles depuis {MANUAL_QA_FILE}...")
        try:
            with open(MANUAL_QA_FILE, "r", encoding="utf-8") as f:
                manual_data = json.load(f)
                for qa in manual_data:
                    dataset.append({
                        "messages": [
                            {"role": "system", "content": "Tu es l'Assistant Expert de la Banque Nationale de Mauritanie (BNM)."},
                            {"role": "user", "content": qa["question"]},
                            {"role": "assistant", "content": qa["answer"]}
                        ]
                    })
            print(f"{len(manual_data)} QA manuelles ajoutées.")
        except Exception as e:
            print(f"Erreur lors du chargement des QA manuelles: {e}")

    # 2. Générer des données synthétiques à partir des documents
    print("Extraction des documents...")
    chunks = process_documents(DATA_DIR)
    print(f"Génération pour {len(chunks)} morceaux...")
    
    for i, chunk in enumerate(chunks):
        qa = generate_qa_pair(chunk["content"])
        if qa:
            dataset.append({
                "messages": [
                    {"role": "system", "content": "Tu es l'Assistant Expert de la Banque Nationale de Mauritanie (BNM)."},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            })
            print(f"[{i+1}/{len(chunks)}] QA générée: {qa['question'][:50]}...")

    # 3. Sauvegarder
    if dataset:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Terminé : {OUTPUT_FILE} contient {len(dataset)} exemples.")
    else:
        print("Aucune donnée disponible.")

if __name__ == "__main__":
    prepare_dataset()
