import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGRetriever:
    def __init__(self, dataset_name, embed_model='all-MiniLM-L6-v2'):
        dataset_dir = os.path.join(os.path.dirname(__file__), '../../dataset', dataset_name)
        rag_index_dir = os.path.join(dataset_dir, 'rag_index')
        self.index = faiss.read_index(os.path.join(rag_index_dir, 'faiss.index'))
        with open(os.path.join(rag_index_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.passages = meta['passages']
        self.meta = meta['meta']
        self.model = SentenceTransformer(embed_model)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            results.append({'passage': self.passages[idx], 'meta': self.meta[idx]})
        return results 