import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bot IA - Expert BNM",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION RAG ---
DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bnm_qa")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DISTANCE_THRESHOLD = 0.70 

# --- STYLING (Final Hybrid: V4 Main + V7 Sidebar + Symmetrical Traits) ---
def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');
        
        /* Main Typography */
        .main-title {
            font-family: 'Playfair Display', serif !important;
            font-size: 3.5rem !important;
            color: #1a2a44 !important;
            margin-bottom: 0px !important;
        }

        .stApp {
            background-color: #ffffff;
        }

        .aqua-accent {
            color: #00bcd4;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 30px;
        }

        /* Sidebar Styling (V7 Style) */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        .status-box {
            background-color: #e6f4ea;
            color: #1e7e34;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            font-family: 'Inter', sans-serif;
            border: 1px solid #c3e6cb;
        }

        /* Message Styling (Symmetrical Traits) */
        /* Hide Avatars */
        [data-testid="stChatMessageAvatarUser"], 
        [data-testid="stChatMessageAvatarAssistant"] {
            display: none !important;
        }

        /* User Message (Natural with Aqua Trait) */
        [data-testid="stChatMessage"]:nth-child(even) {
            background-color: transparent !important;
            border: none !important;
            border-left: 4px solid #00bcd4 !important; /* The user 'trait' */
            padding: 0.5rem 0 0.5rem 1.5rem !important;
            margin-bottom: 2rem !important;
        }

        /* Assistant Message (Natural with Navy Trait) */
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: transparent !important;
            border: none !important;
            border-left: 4px solid #1a2a44 !important; /* The assistant 'trait' */
            padding: 0.5rem 0 0.5rem 1.5rem !important;
            margin-bottom: 2rem !important;
            color: #31333f !important;
        }

        /* Sidebar Buttons */
        .stButton>button {
            background-color: #ffffff;
            border: 1px solid #d1d5db;
            color: #374151;
            border-radius: 8px;
            width: 100%;
            font-family: 'Inter', sans-serif;
        }
        
        .stButton>button:hover {
            border-color: #1a2a44;
            color: #1a2a44;
        }

        /* Input Bar */
        .stChatInputContainer {
            border-top: 1px solid #e5e7eb !important;
            background-color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# --- UTILS ---
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)

@st.cache_resource
def get_embedding_model():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

client_ollama = ollama.Client(host=OLLAMA_HOST)

def get_rag_context(query):
    try:
        if not os.path.exists(DB_PATH):
            return None, None
            
        client = get_chroma_client()
        embedding_function = get_embedding_model()
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        
        # Increase n_results for better context coverage
        results = collection.query(query_texts=[query], n_results=5)
        
        if results['documents'] and len(results['documents'][0]) > 0:
            relevant_docs = []
            sources = []
            for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                if dist < DISTANCE_THRESHOLD:
                    relevant_docs.append(doc)
                    sources.append(meta.get("source", "Inconnu"))
            
            if relevant_docs:
                return "\n---\n".join(relevant_docs), list(set(sources))
            
    except Exception:
        pass
    return None, None

# --- UI CONTENT ---
# Sidebar (Restoring V7 Functional Style)
with st.sidebar:
    st.title("État du système")
    st.markdown('<div class="status-box">Ollama est prêt.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Modèle")
    model_choice = st.selectbox("Sélectionner un modèle", ["qwen2.5:1.5b", "qwen2.5:7b"], label_visibility="collapsed")
    
<<<<<<< HEAD
    # Check if model is available
    try:
        models_info = client_ollama.list()
        print(f"DEBUG: client_ollama.list() type: {type(models_info)}", flush=True)
        print(f"DEBUG: client_ollama.list() content: {models_info}", flush=True)
        
        # Handle response being either a dict or an object
        if isinstance(models_info, dict):
             models_list = models_info.get('models', [])
        else:
             models_list = getattr(models_info, 'models', [])

        available_models = []
        for m in models_list:
            # Handle item being either a dict or an object
            if isinstance(m, dict):
                model_name = m.get('name') or m.get('model')
            else:
                model_name = getattr(m, 'name', getattr(m, 'model', None))
            
            if model_name:
                available_models.append(model_name)
        
        print(f"DEBUG: Available models found: {available_models}", flush=True)
        
        # Check for partial matches (e.g. qwen2.5:7b:latest)
        is_model_available = any(model_choice in m for m in available_models)
        
        if not is_model_available:
            st.warning(f"⚠️ Le modèle '{model_choice}' n'est pas encore téléchargé sur le serveur via Ollama. Le service 'ollama-init' est peut-être encore en cours d'exécution.")
            print(f"DEBUG: Warning user that model {model_choice} is missing.", flush=True)
            
    except Exception as e:
        # Fallback: if checking fails, don't crash, just log/warn nicely
        print(f"ERROR: Failed to check models: {e}", flush=True)
        # st.error(f"Attention: Impossible de vérifier les modèles disponibles ({e})")
        pass

=======
    st.markdown("---")
>>>>>>> 6ea13b4 (Ajout de certaines modifications)
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# Main Body (Luxury V4 Style Content)
st.markdown('<h1 class="main-title">Bot IA - Expert BNM</h1>', unsafe_allow_html=True)
st.markdown('<p class="aqua-accent">Service client intelligent de la Banque Nationale de Mauritanie.</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Logic
if prompt := st.chat_input("Posez votre question sur la BNM..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context, sources = get_rag_context(prompt)
        
        # --- HARDENED SYSTEM PROMPT ---
        system_prompt = f"""Tu es l'Expert IA exclusif de la Banque Nationale de Mauritanie (BNM).
IDENTITÉ : Ton organisation est la Banque Nationale de Mauritanie (Mauritanie), et non le Bureau National des Monnaies d'Afrique de l'Ouest.
MISSION : Répondre aux questions des clients sur les services, produits et régulations de la BNM Mauritanie.

RÈGLES STRICTES :
1. UTILISE UNIQUEMENT LE CONTEXTE CI-DESSOUS. 
2. Si l'information ne figure pas dans le contexte, dis : "Je suis désolé, mais je n'ai pas d'information officielle à ce sujet dans mes documents."
3. Ne fais JAMAIS de suppositions sur les régulations gouvernementales ou les normes techniques (comme le crédit documentaire) si elles ne sont pas explicitement décrites dans le contexte.
4. Ton doit être professionnel, précis et corporatif.

CONTEXTE DOCUMENTAIRE :
{context if context else 'AUCUN DOCUMENT DISPONIBLE. Refuse de répondre.'}
"""

        try:
            # RÈGLE : STREAMING ACTIVÉ
            response_stream = client_ollama.chat(
                model=model_choice,
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages[-3:],
                stream=True
            )
            
            def generate_stream():
                for chunk in response_stream:
                    yield chunk['message']['content']
            
            full_response = st.write_stream(generate_stream())
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if sources:
                st.caption(f"Sources : {', '.join(sources)}")
            
        except Exception as e:
            st.error(f"Incident technique : {e}")
