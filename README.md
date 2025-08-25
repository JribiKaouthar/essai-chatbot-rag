# ESSAI Chatbot (RAG) with TinyLlama and Streamlit

An intelligent chatbot designed for the **Higher School of Statistics and Information Analysis (ESSAI) in Tunis**.  
The chatbot leverages **Retrieval-Augmented Generation (RAG)** to answer user questions from institutional PDF documents.  
It uses **FAISS semantic search** with MiniLM embeddings for document retrieval, and **TinyLlama** as the local language model, all deployed inside a **Streamlit web interface**.

---

## 🚀 Features
- 📄 **PDF ingestion** – load ESSAI documents with `PyPDFLoader`
- ✂️ **Text chunking** – split documents into manageable sections with LangChain
- 🔍 **Vector search** – FAISS index with `sentence-transformers/all-MiniLM-L6-v2`
- 🤖 **TinyLlama model** – local causal LM for response generation
- 🧠 **RAG pipeline** – combine context retrieval + generative model
- 💬 **Streamlit UI** – clean chat interface with styled user and bot bubbles
- ⚡ **Optimized for CPU** – runs on local machines without GPUs
- 🔒 **.env configuration** – easy setup without exposing secrets

---

## 📂 Project Structure
essai-chatbot-rag/
│── app.py # Main Streamlit app
│── requirements.txt # Python dependencies
│── .env.example # Example environment variables
│── README.md # Project documentation
│── .gitignore # Files and folders to ignore
│
├── src/ # (Optional) extra Python modules
├── data/ # Place PDFs here (not committed)
│ └── README.md
├── assets/ # Images, logos, icons
│ └── essai-logo.ico


---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/<your-username>/essai-chatbot-rag.git
cd essai-chatbot-rag

2. Create a virtual environment
python -m venv .venv
# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables

Copy .env.example to .env:

cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows


Edit .env with your actual paths (PDF folder, model folder, etc.).

📚 Models & Data

PDF documents → place them in the data/ folder.

Language model (TinyLlama) → download from Hugging Face and place in ./TinyLlama.
Example:

git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama

▶️ Run the Application

Start the chatbot locally:

streamlit run app.py


Then open your browser at http://localhost:8501
.

📊 Example Workflow

User asks: “What is ESSAI?”

FAISS retrieves the most relevant sections from PDF documents.

TinyLlama generates a contextualized answer.

The answer is displayed in the chat interface with a friendly style.

🛠️ Technologies Used

Python

Streamlit

LangChain

FAISS

Hugging Face Transformers

TinyLlama

📌 Future Improvements

Deploy on the web (Streamlit Cloud or Hugging Face Spaces)

Add multilingual support (Arabic / French / English)

Enhance with larger LLMs (LLaMA 2, Mistral, etc.)

Integrate authentication for student/teacher access

📝 License

This project is licensed under the MIT License – feel free to use and adapt.

📧 Contact

Kaouthar Jribi
📍 Tunis, Tunisia
📩 jribikaouthar@yahoo.com

🔗 linkedin.com/in/kaouthar-jribi


---

