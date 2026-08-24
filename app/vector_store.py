import chromadb
from sentence_transformers import SentenceTransformer


class VectorStoreManager:
    """
    Manages vector embeddings and ChromaDB storage for OmniBrain-RAG.
    """

    def __init__(
        self,
        collection_name: str = "omnibrain_docs",
        db_path: str = "data/chroma_db"
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # Load SentenceTransformer embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize persistent ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(self, chunks: list[dict]):
        """
        Convert document chunks into embeddings and store them in ChromaDB.
        """
        if not chunks:
            return

        documents = [chunk["text"] for chunk in chunks]

        metadatas = [
            {
                "page": chunk["page"],
                "source": chunk["source"]
            }
            for chunk in chunks
        ]

        ids = [
            f"{chunk['source']}_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()

        # Store in ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def add_documents(self, chunks: list[dict]):
        """
        Alias for add_chunks().
        """
        return self.add_chunks(chunks)

    def search_similar(self, query: str, top_k: int = 3):
        """
        Search ChromaDB for chunks most similar to the query.
        """
        if not query:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        return results

    def count(self):
        """
        Return the number of stored chunks.
        """
        return self.collection.count()


if __name__ == "__main__":
    print("Vector Store Manager module created successfully.")