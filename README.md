# 💬 MSME Schemes Chatbot (GenAI-Powered)

A **Generative AI-enabled chatbot** developed at **Wavenet Solutions Pvt. Ltd.** to simplify the discovery and understanding of MSME schemes. This tool helps users navigate government schemes by reducing manual effort in deciphering eligibility criteria and streamlining the application process.

---

## 🚀 Features

- ✅ Conversational chatbot interface using Streamlit
- 🔍 RAG (Retrieval-Augmented Generation) architecture
- 🤖 Fine-tuned LLM (Mistral-7B-Instruct) for accurate and relevant answers
- 📚 Intelligent semantic search using FAISS and Sentence Transformers
- 🧠 Feedback logging for continual improvement

---

## 🛠 Functional Requirements & Architecture

### 1. Model Training
- **Dataset Preparation**: Includes MSME scheme documents, FAQs, and policy documents from central/state government.
- **LLM Fine-tuning**: Mistral-7B-Instruct model used with domain-specific prompts and documents.
- **Prompt Engineering**: Ensures relevant and accurate query resolution using structured prompts.

### 2. RAG-Based Architecture
- **Data Ingestion & Indexing**: Scheme documents split into chunks using LangChain and embedded via Sentence Transformers into FAISS.
- **Contextual Retrieval**: Relevant text chunks retrieved based on semantic similarity to user queries.
- **LLM Integration**: Retrieved context is fed to the LLM for factual and personalized responses.
- **Feedback Loop**: User reactions (👍 / 👎) are logged to refine future outputs.

---

## 🧩 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Text Splitting**: [LangChain](https://www.langchain.com/)
- **Language**: Python 3.x

---

## 📁 Folder Structure


├── app.py # Streamlit-based web UI
├── llm_agent.py # LLM initialization and query response logic
├── vector_store.py # FAISS index creation with sentence embeddings
├── data_loader.py # JSON parser and text chunker
├── README.md # Project documentation (this file)



---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/msme-genai-chatbot.git
cd msme-genai-chatbot


Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the application

bash
Copy
Edit
streamlit run app.py

📂 Dataset
You can download the sample dataset used for training from this link:
👉 Download Dataset[Dataset](https://we.tl/t-GLtJRoUoYq)

