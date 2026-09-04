import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from langchain_ollama import ChatOllama

load_dotenv(override=True)

class RAGSearch:

    def __init__(self, persist_directory: str = "../data/faiss_store", embedding_model_name: str = "qwen3-embedding", chunk_size: int = 1000, chunk_overlap: int = 200, batch_size: int = 100):
        self.vector_store = FaissVectorStore(persist_directory, embedding_model_name, chunk_size, chunk_overlap, batch_size)
        #Load or build the vector store
        faiss_path = os.path.join(persist_directory, "faiss.index")
        metadata_path = os.path.join(persist_directory, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(metadata_path)):
            print(f"[INFO] FAISS index or metadata not found. Building vector store...")
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            print(f"[INFO] Loading existing FAISS index and metadata...")
            self.vector_store.load()

        self.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))
        print(f"[INFO] Loaded LLM model: {os.getenv('OLLAMA_MODEL')}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Search the vector store
        print(f"[INFO] Searching for top {top_k} documents for query: '{query}'")
        results = self.vector_store.search(query, top_k)
        texts = [res['metadata'].get('text', '') for res in results if res['metadata']]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        # Summarize the results using the LLM
        prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}"
        summary = self.llm.invoke([prompt])
        return summary.content