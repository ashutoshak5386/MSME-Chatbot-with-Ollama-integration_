import streamlit as st
import os
import ollama
import numpy as np
from data_loader import load_schemes_json, load_all_schemes_from_folder
from vector_store import build_faiss_index

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': False,
        'history': [],
        'model_name': 'mistral',  # Default model
        'ollama_running': False
    })

# Page configuration
st.set_page_config(page_title="MSME Schemes Chatbot", layout="wide")
st.title("MSME Schemes Chatbot")

# Verify Ollama connection
try:
    # Quick ping to check if Ollama is running
    ollama.list()
    st.session_state.ollama_running = True
except Exception:
    st.error("Ollama service not running. Please start Ollama first with:")
    st.code("ollama serve")
    st.stop()

# Document processing
if not st.session_state.initialized:
    data_folder = "data"
    if os.path.exists(data_folder):
        all_chunks = load_all_schemes_from_folder(data_folder)
        if all_chunks:
            index, embedder = build_faiss_index(all_chunks)
            st.session_state.update({
                'chunks': all_chunks,
                'index': index,
                'embedder': embedder,
                'initialized': True
            })
    
    # File uploader for additional documents
    uploaded = st.file_uploader("Upload scheme JSON", type=['json'])
    if uploaded:
        try:
            new_chunks = load_schemes_json(uploaded)
            st.session_state.chunks += new_chunks
            st.session_state.index, st.session_state.embedder = build_faiss_index(st.session_state.chunks)
            st.success("Document indexed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Model selection
available_models = ['mistral', 'llama3']
st.session_state.model_name = st.selectbox(
    "Select Model",
    available_models,
    index=available_models.index(st.session_state.model_name)
)

# Chat interface
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input("Ask about MSME schemes:"):
    if not st.session_state.initialized:
        st.warning("Please upload scheme documents first")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Retrieve relevant context
            q_emb = st.session_state.embedder.encode([prompt])
            _, idxs = st.session_state.index.search(np.array(q_emb, dtype='float32'), 3)
            context = "\n".join([st.session_state.chunks[i] for i in idxs[0] if i < len(st.session_state.chunks)])
            
            # Generate response
            messages = [
                {"role": "system", "content": "You are an expert on MSME schemes. Answer concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
            ]
            
            try:
                for chunk in ollama.chat(
                    model=st.session_state.model_name,
                    messages=messages,
                    stream=True
                ):
                    if chunk['message']['content']:
                        full_response += chunk['message']['content']
                        response_placeholder.markdown(full_response)
                
                st.session_state.history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Feedback system
if st.session_state.history and st.session_state.history[-1]['role'] == 'assistant':
    latest = st.session_state.history[-1]['content']
    col1, col2 = st.columns(2)
    if col1.button("👍"):
        with open("feedback.log", "a") as f:
            f.write(f"POSITIVE: {latest}\n")
        st.toast("Feedback recorded!")
    if col2.button("👎"):
        with open("feedback.log", "a") as f:
            f.write(f"NEGATIVE: {latest}\n")
        st.toast("Feedback recorded!")