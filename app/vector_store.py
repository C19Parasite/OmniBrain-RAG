import os
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    """
    Manages vector embeddings and ChromaDB storage for OmniBrain-RAG.
    """
    def __init__(self, collection_name: str = "omnibrain_docs", db_path: str = "data/chroma_db"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[dict]):
        """
        Converts text chunks into embeddings and saves them to ChromaDB.
        """
        if not chunks:
            return

        documents = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"], "source": c["source"]} for c in chunks]
        ids = [f"chunk_{c['chunk_id']}" for c in chunks]

        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()

        # Add to Chroma DB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        def add_documents(self, chunks):
         """Alias for add_chunks to support alternate method calls."""
        return self.add_chunks(chunks)

    def search_similar(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Queries ChromaDB for the most relevant chunks matching a text query.
        """
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results

if __name__ == "__main__":
    print("Vector Store Manager module created successfully.")
