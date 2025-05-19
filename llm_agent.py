import ollama
import numpy as np
from typing import Optional, Callable

def initialize_llm(model_name: str = "mistral"):
    """
    Initialize connection to local Ollama model.
    """
    try:
        # Check if model is available, pull if not
        available_models = [model['name'] for model in ollama.list()['models']]
        if model_name not in available_models:
            ollama.pull(model_name)
        return model_name
    except Exception as e:
        raise Exception(f"Failed to initialize Ollama model: {str(e)}")

def answer_query(
    query: str,
    index,
    embedder,
    chunks,
    model_name: str,
    top_k: int = 3,
    stream_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Generate answer using Ollama's local model with RAG
    """
    try:
        # Retrieve relevant chunks
        q_emb = embedder.encode([query])
        _, idxs = index.search(np.array(q_emb, dtype='float32'), top_k)
        retrieved = [chunks[i] for i in idxs[0] if i < len(chunks)]
        
        if not retrieved:
            return "No relevant information found in the documents."
            
        context = "\n".join(retrieved)
        
        # Create prompt template based on model
        if "llama3" in model_name.lower():
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for MSME schemes. Answer questions using ONLY the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
        else:  # Mistral and others
            messages = [
                {
                    "role": "user",
                    "content": f"""<s>[INST] Answer the question using ONLY the provided context.
Context: {context}
Question: {query} [/INST]"""
                }
            ]
        
        # Generate response
        if stream_callback:
            # Stream the response
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            )
            
            full_response = ""
            for chunk in stream:
                part = chunk['message']['content']
                full_response += part
                stream_callback(full_response)
            
            return full_response
        else:
            # Single response
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            )
            return response['message']['content']
    except Exception as e:
        return f"Error generating answer: {str(e)}"