from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from src.data_loader import load_all_documents
import ollama

class EmbeddingPipeline:

    def __init__(self, model_name: str="qwen3-embedding", chunk_size: int = 1000, chunk_overlap: int = 200, batch_size = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = None
        self.batch_size = batch_size
        self._embedding_dimension = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = ollama.embeddings(
                model = self.model_name,
                prompt = "Test connection"
            )
    
            self._embedding_dimension = len(self.model["embedding"])
            print(f" ✅ connected to ollama model: {self.model_name}")
            print(f" ✅ Embedding Dimension: {self._embedding_dimension}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running a model: {self.model_name}")
            raise

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not chunks:
            raise ValueError("Chunk List is empty")

        print(f"[INFO] Generating embedding for {len(chunks)} chunks...")
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embedding = []
            for text in batch:
                try:
                    response = ollama.embeddings(
                        model=self.model_name,
                        prompt=text.page_content
                    )
                    batch_embedding.append(response["embedding"])
                except Exception as e:
                    print(f"❌ Error embedding text: {e}")
                    raise

            embeddings.extend(batch_embedding)
            
            # Progress indicator
            processed = min(i+self.batch_size, len(chunks))
            print(f" Progress: {processed}/{len(chunks)} texts processed")
            
        embeddings_array = np.array(embeddings)
        print(f"✅ Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
            
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model"""
        if self._embedding_dimension is None:
            raise ValueError("Embedding model not initialized properly")
        return self._embedding_dimension
        