from rca.baseline.rag_retriever import RAGRetriever
from rca.api_router import get_chat_completion

class RAGAgent:
    def __init__(self, dataset_name, prompt_schema, top_k=5):
        self.retriever = RAGRetriever(dataset_name)
        self.prompt_schema = prompt_schema  # This should include cand and clarifications
        self.top_k = top_k

    def run(self, query, logger, top_k=None):
        k = top_k if top_k is not None else self.top_k
        retrieved = self.retriever.retrieve(query, top_k=k)
        context = '\n\n'.join([r['passage'] for r in retrieved])
        prompt = f"""You are a Root Cause Analysis (RCA) assistant.\n\nContext:\n{context}\n\n{self.prompt_schema}\n\nQuestion: {query}\n\nAnswer using only the above context."""
        logger.info(f"Prompt to LLM:\n{prompt}")
        answer = get_chat_completion([
            {'role': 'system', 'content': prompt}
        ])
        logger.info(f"LLM Answer: {answer}")
        return answer, retrieved 