import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def build_faiss_index(chunks):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index, embedder