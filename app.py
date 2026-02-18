import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os
import re

# Page Config
st.set_page_config(page_title="IA Conversationnelle - BNM Expert", layout="wide")

# Configuration RAG
DB_PATH = r"c:\Users\Lenovo\.gemini\antigravity\playground\harmonic-pioneer\chroma_db"
COLLECTION_NAME = "bnm_qa"
# Seuil de distance (plus c'est bas, plus c'est proche).
# 0.8 - 1.0 est souvent un bon compromis pour sentence-transformers.
DISTANCE_THRESHOLD = 0.95 

def is_greeting(text):
    greetings = [
        "salut", "bonjour", "bonsoir", "hello", "hi", "hey", 
        "سلام", "صباح الخير", "مساء الخير", "مرحبا"
    ]
    # Simple regex to check if the first words contain a greeting
    clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return any(clean_text.startswith(g) for g in greetings) or len(clean_text) < 3

def get_rag_context(query):
    try:
        if not os.path.exists(DB_PATH):
            return None, None
            
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        
        # On demande un peu plus de résultats pour plus de contexte
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            best_distance = results['distances'][0][0]
            if best_distance < DISTANCE_THRESHOLD:
                # On concatène les documents pertinents
                relevant_docs = []
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    if dist < DISTANCE_THRESHOLD:
                        relevant_docs.append(doc)
                return "\n---\n".join(relevant_docs), best_distance
            
    except Exception as e:
        # Erreur silencieuse pour l'utilisateur, loguée en console si besoin
        pass
    return None, None

st.title("Bot IA - Expert BNM")
st.markdown("Service client intelligent de la **Banque Nationale de Mauritanie**.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("État du système")
    try:
        ollama.list()
        st.success("Ollama est prêt.")
    except Exception:
        st.error("Ollama n'est pas détecté.")
        st.stop()
    
    st.divider()
    model_choice = st.selectbox("Modèle", ["qwen2.5:7b", "qwen2.5:1.5b", "qwen2:1.5b"], index=0)
    
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Posez votre question sur la BNM..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval process (No UI Status)
    context, distance = get_rag_context(prompt)
    greeting_detected = is_greeting(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # SYSTEM PROMPT strict
        system_prompt = """Tu es l'Assistant Expert de la Banque Nationale de Mauritanie (BNM).

TES RÈGLES CRITIQUES :
1. TON DE VOIX : Professionnel, accueillant et institutionnel.
2. DISCIPLINE LINGUISTIQUE : Réponds EXCLUSIVEMENT dans la langue utilisée par l'utilisateur (Français, Anglais ou Arabe). Ne mélange JAMAIS deux langues dans la même réponse.
3. RESTRICTION DE DOMAINE : Tu ne traites QUE les questions liées à la BNM (produits, services, tarifs, agences).
4. RÉPONSES EXACTES : Utilise le contexte fourni pour donner des réponses précises. Si l'information n'est pas dans le contexte, décline poliment en expliquant que tu es spécialisé uniquement dans les services bancaires de la BNM.
5. HORS-SUJET : Si la question n'a rien à voir avec la banque, réponds : "Désolé, je suis un assistant spécialisé uniquement dans les services de la BNM. Je ne peux pas répondre à cette demande."
6. SALUTATIONS : Si l'utilisateur te salue, salue-le en retour cordialement et demande-lui comment tu peux l'aider concernant les services de la BNM.

CONTEXTE BNM FOURNI :
{context}
"""
        # Formattage du prompt système avec le contexte
        formatted_system = system_prompt.format(context=context if context else "Aucun contexte spécifique trouvé.")

        messages_to_send = [{"role": "system", "content": formatted_system}]
        
        # Ajout de l'historique récent (limité pour éviter la confusion)
        messages_to_send.extend(st.session_state.messages[-5:-1])
        
        # Dernier message utilisateur
        if greeting_detected:
            # On force une instruction de salutation si c'est détecté
            messages_to_send.append({"role": "user", "content": f"[SALUTATION DÉTECTÉE] {prompt}"})
        else:
            messages_to_send.append({"role": "user", "content": prompt})

        try:
            responses = ollama.chat(
                model=model_choice,
                messages=messages_to_send,
                stream=True
            )
            
            for chunk in responses:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")

