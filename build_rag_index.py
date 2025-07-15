import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Usage: python build_rag_index.py <DATASET_NAME>

CHUNK_SIZE = 20  # Number of rows per chunk for tabular data
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Helper to chunk a dataframe into text passages
def chunk_dataframe(df, file_name, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        text = f"File: {file_name}\n" + chunk.to_csv(index=False)
        chunks.append(text)
    return chunks

def main(dataset_name):
    dataset_dir = os.path.join(os.path.dirname(__file__), dataset_name)
    telemetry_dir = os.path.join(dataset_dir, 'telemetry')
    rag_index_dir = os.path.join(dataset_dir, 'rag_index')
    os.makedirs(rag_index_dir, exist_ok=True)
    passages = []
    meta = []

    # Add clarifications and cand from prompt file if available
    for prompt_file in [
        'rca/baseline/rca_agent/prompt/basic_prompt_' + dataset_name.split('/')[0].capitalize() + '.py',
        'rca/baseline/rca_agent/prompt/agent_prompt.py',
    ]:
        try:
            with open(os.path.join(os.path.dirname(__file__), '../../', prompt_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'cand =' in content:
                    cand = content.split('cand =',1)[1].split('"""',2)[1]
                    passages.append(cand)
                    meta.append({'type': 'cand', 'source': prompt_file})
                if '_clarification =' in content:
                    clar = content.split('_clarification =',1)[1].split('"""',2)[1]
                    passages.append(clar)
                    meta.append({'type': 'clarification', 'source': prompt_file})
        except Exception:
            pass

    # Index all CSVs in telemetry, record, query
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.csv'):
                file_path = os.path.join(root, f)
                try:
                    df = pd.read_csv(file_path)
                    chunks = chunk_dataframe(df, f)
                    passages.extend(chunks)
                    meta.extend([{'type': 'csv', 'file': f, 'chunk': i} for i in range(len(chunks))])
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Total passages: {len(passages)}")
    # Embed passages
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index and mapping
    faiss.write_index(index, os.path.join(rag_index_dir, 'faiss.index'))
    with open(os.path.join(rag_index_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'passages': passages, 'meta': meta}, f)
    print(f"RAG index built and saved to {rag_index_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python build_rag_index.py <DATASET_NAME>")
        sys.exit(1)
    main(sys.argv[1]) 