import os
import faiss
import numpy as np
import pickle
import ollama
from typing import List, Any

from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_directory: str = "../data/faiss_store", embedding_model_name: str = "qwen3-embedding", chunk_size: int = 1000, chunk_overlap: int = 200, batch_size: int = 100):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        print(f"Initialized FaissVectorStore with model: {self.embedding_model}, chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}, batch_size: {self.batch_size}")

    def build_from_documents(self, documents: List[Any]):
        print(f"Bulding FAISS vector store from {len(documents)} documents...")
        embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, batch_size=self.batch_size)
        chunks = embedding_pipeline.chunk_documents(documents)
        embeddings = embedding_pipeline.embed_chunks(chunks)
        metadatas = [{"text" : chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_directory}")


    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to FAISS index. Total embeddings: {self.index.ntotal}")


    def save(self):
        faiss_path = os.path.join(self.persist_directory, "faiss.index")
        metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] FAISS index saved to {faiss_path} and metadata saved to {metadata_path}")


    def load(self):
        faiss_path = os.path.join(self.persist_directory, "faiss.index")
        metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(faiss_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] FAISS index loaded from {faiss_path} and metadata loaded from {metadata_path}")
        else:
            print(f"[WARN] FAISS index or metadata not found in {self.persist_directory}. Please build the vector store first.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying FAISS vector store with text: {query_text}")
        response = ollama.embeddings(model=self.embedding_model, prompt=query_text)
        query_embedding = np.array([response["embedding"]], dtype="float32")
        results = self.search(query_embedding, top_k)
        print(f"[INFO] Retrieved {len(results)} results from FAISS vector store")
        return results


