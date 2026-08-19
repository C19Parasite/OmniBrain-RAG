DOCUMENTS = []

def add_document(filename: str, text: str):
    DOCUMENTS.append({
        "filename": filename,
        "text": text
    })
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

    def add_documents(self, file_path, chunks):
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