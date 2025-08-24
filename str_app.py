import streamlit as st
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import warnings
import re
import time
import logging



# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings("ignore")

# Set device to CPU to avoid out of memory errors
device = "cpu"

# Initialize conversation history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def main():
    st.set_page_config(page_title="ESSAI Chatbot", page_icon="C:\\Users\\kaout\\Desktop\\med2\\essai-logo.ico", layout="wide")

    @st.cache_resource
    def download_hugging_face_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embeddings = download_hugging_face_embeddings()

    @st.cache_resource
    def load_pdf_documents(_data_folder="data"):
        all_documents = []
        for filename in os.listdir(_data_folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(_data_folder, filename)
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
        return all_documents

    documents = load_pdf_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    @st.cache_resource
    def create_faiss_index(_embedding_model, _documents):
        texts = [doc.page_content for doc in _documents]
        return FAISS.from_texts(texts, _embedding_model)

    faiss_index = create_faiss_index(embeddings, text_chunks)

    logging.basicConfig(level=logging.INFO)

    @st.cache_resource
    def load_model():
        print("Démarrage du chargement du modèle...")
        start_time = time.time()
        model_directory = "C:\\Users\\kaout\\Desktop\\med2\\TinyLlama"  
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_directory)
            model = AutoModelForCausalLM.from_pretrained(model_directory)
            model.to(device)
            logging.info(f"Modèle chargé en {time.time() - start_time:.2f} secondes")
            return tokenizer, model
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle : {e}")
            raise e

    tokenizer, model = load_model()

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=None,
        temperature=0.6,
        max_new_tokens=100,
        do_sample=True,
        device=device
    )

    llm_pipeline_instance = HuggingFacePipeline(pipeline=llm_pipeline)

    def retrieve_context(question):
        try:
            relevant_docs = faiss_index.similarity_search(question, k=3)
            context = " ".join([doc.page_content[:300] for doc in relevant_docs])  # Limiter la taille
            return context
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du contexte : {e}")
            return ""


    def generate_prompt(question, context):
        return f"""
        Vous êtes un assistant intelligent, bienveillant et compétent, conçu pour aider les visiteurs du site de l'ESSAI en leur fournissant des réponses précises et empathiques.

        Question de l'utilisateur : "{question}"

        Contexte : "{context}"

        Instructions :
        - Fournissez une réponse concise et informative, adaptée au contexte.
        - Utilisez un ton bienveillant et engageant pour rendre l'interaction agréable.
        - Soyez pertinent et assurez-vous que l'utilisateur a toutes les informations nécessaires pour comprendre et avancer.

        
        Réponse :
        """


        

    def custom_qa_chain(question):
        try:
            context = retrieve_context(question)
            if not context:
                return "Je n'ai pas pu trouver d'informations pertinentes dans les documents. Pouvez-vous préciser votre question ?"
            prompt = generate_prompt(question, context)
            print(prompt)
            st.write(f"Prompt généré : {prompt}") 
            # Ensure truncation is set during tokenization
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500).input_ids.to(device)

            result = llm_pipeline_instance(prompt)
            
            if isinstance(result, list) and len(result) > 0:
                # Clean up any stray HTML tags
                response = result[0]['generated_text'] if 'generated_text' in result[0] else result[0]
                st.write("Réponse générée :", response)
                match = re.search(r'Response:\s*(.*)', response, re.DOTALL)
                if match:
                    remarque="ok match"
                    final_response = match.group(1).strip()
                    if len(final_response) < 1000:
                        final_response += " N'hésitez pas à demander si vous avez besoin de plus d'informations ou de clarification !"
                    return final_response
            return "Je suis désolé, je n'ai pas pu trouver d'informations pertinentes."
        except Exception as e:
            logging.error(f"Erreur dans la chaîne de questions-réponses : {e}")
            return f"Erreur dans la chaîne de questions-réponses : {e}"

    def handle_greetings(user_input):
        greetings = ["Salut", "Bonjour", "Hey", "Bonjour", "Bon après-midi", "Bonsoir"]
        if any(greeting in user_input.lower() for greeting in greetings):
            return ("Bonjour ! 😊 Comment puis-je vous aider avec vos questions sur l'ESSAI aujourd'hui ? "
                    "N'hésitez pas à demander des informations sur l'ESSAI.")
        return None

    # CSS styling to make the interface use the full width and remove the white rectangle issue
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #95a2bd;
            width: 100%;
            margin: 0;
            padding: 0;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding: 20px;
            width: 100%;
            margin: 0 auto;
        }
        .chat-bubble {
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 16px;
        }
        .user-bubble {
            background-color: #e0f7fa;
            align-self: flex-end;
            color: #00796b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .bot-bubble {
            background-color: #fbe9e7;
            align-self: flex-start;
            color: #d32f2f;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .input-container {
            display: flex;
            justify-content: center;
            width: 100%;
            margin: 20px auto;
        }
        .input-box {
            flex-grow: 1;
            margin-right: 10px;
            padding: 12px;
            font-size: 16px;
            border-radius: 20px;
            border: 1px solid #ccc;
        }
        .submit-button {
            padding: 12px;
            font-size: 16px;
            background-color: #00796b;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 20px;
        }
        .submit-button:hover {
            background-color: #004d40;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    st.title("ESSAI Chatbot ")
    st.write("Demandez-moi n'importe quoi sur l'ESSAI et je vous aiderai!")

    # Display the conversation so far
    st.write("### Historique des discussions")
    with st.container():
        for i, (message, is_user) in enumerate(st.session_state.conversation):
            if is_user:
                st.markdown(f'<div class="chat-bubble user-bubble">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message}</div>', unsafe_allow_html=True)

    # Input field for new messages
    st.write(" ### Tapez votre message")
    with st.container():
        user_input = st.text_input("You:", key="user_input_key")

    # Submission button
    if  st.button("Envoyer", key="submit_key"):
        if user_input:
            st.session_state.conversation.append((user_input, True))

            greeting_response = handle_greetings(user_input)
            if greeting_response:
                bot_response = greeting_response
            else:
                bot_response = custom_qa_chain(user_input)

            st.session_state.conversation.append((bot_response, False))

if __name__ == "__main__":
    main()
