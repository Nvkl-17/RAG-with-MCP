import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp import MCPMessage

class IngestionAgent:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def process(self, raw_text, sender="IngestionAgent", receiver="RetrievalAgent"):
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
        embeddings = self.model.encode(chunks)
        return MCPMessage(sender, receiver, "EMBEDDINGS_READY", {
            "chunks": chunks,
            "embeddings": embeddings.tolist()
        }).get()

class RetrievalAgent:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def receive(self, msg):
        if msg["type"] == "EMBEDDINGS_READY":
            embs = np.array(msg["payload"]["embeddings"])
            self.chunks = msg["payload"]["chunks"]
            self.index = faiss.IndexFlatIP(embs.shape[1])
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)  # normalize embeddings
            self.index.add(embs)

    def retrieve(self, query, top_k=3, sender="RetrievalAgent", receiver="LLMResponseAgent"):
        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        D, I = self.index.search(query_emb, top_k)
        top_chunks = [self.chunks[i] for i in I[0]]
        return MCPMessage(sender, receiver, "CONTEXT_RESPONSE", {
            "top_chunks": top_chunks,
            "query": query
        }).get()

class LLMResponseAgent:
    def __init__(self):
        pass

    def generate(self, msg):
        ctx = "\n".join(msg["payload"]["top_chunks"])
        query = msg["payload"]["query"]
        response = f"🤖 Based on the document, the answer to your question:\n**{query}**\nis:\n➡️ {ctx[:300]}..."
        return response
