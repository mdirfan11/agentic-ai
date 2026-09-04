from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    #docs = load_all_documents("data")
    vector_store = FaissVectorStore("faiss_store")
    #vector_store.build_from_documents(docs)
    vector_store.load()
    print(vector_store.query("What is multithreading in java?", top_k=5))

    rag_search = RAGSearch()
    query = "What is multithreading in java?"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print(f"Summary for query '{query}':\n{summary}")