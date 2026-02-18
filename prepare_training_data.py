import pandas as pd
import json
import os

EXCEL_PATH = r"C:\Users\Lenovo\OneDrive\Documents\Docs Bnm QR.xlsx"
OUTPUT_PATH = "bnm_dataset.jsonl"

def prepare():
    print(f"Reading Excel file: {EXCEL_PATH}")
    try:
        df = pd.read_excel(EXCEL_PATH)
        
        # Robust column detection (same as ingest_data.py)
        question_col = None
        answer_col = None
        for col in df.columns:
            clean_col = str(col).replace('\x00', '').strip().lower()
            if 'question' in clean_col:
                question_col = col
            elif 'ponse' in clean_col:
                answer_col = col
        
        if not question_col or not answer_col:
            print(f"Error identification columns: {df.columns.tolist()}")
            return

        dataset = []
        system_msg = "Tu es un assistant expert de la BNM (Banque Nationale de Mauritanie). Réponds de manière précise et professionnelle."

        for _, row in df.iterrows():
            question = str(row[question_col])
            answer = str(row[answer_col])
            
            # Format ChatML / OpenAI Messages
            entry = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            dataset.append(entry)

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Successfully created {len(dataset)} examples in {OUTPUT_PATH}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prepare()
