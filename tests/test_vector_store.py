from app.vector_store import VectorStoreManager


def main():
    vector_store = VectorStoreManager()

    chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "text": "Revenue increased by 25 percent during the financial year.",
            "source": "test.pdf",
        },
        {
            "chunk_id": 1,
            "page": 2,
            "text": "The company expanded its operations into three new markets.",
            "source": "test.pdf",
        },
        {
            "chunk_id": 2,
            "page": 3,
            "text": "Operating expenses decreased compared with the previous year.",
            "source": "test.pdf",
        },
    ]

    print("Adding documents...")

    vector_store.add_documents(chunks)

    print("Documents added successfully.")

    query = "What happened to the company's revenue?"

    print(f"\nQuery: {query}")

    results = vector_store.search_similar(
        query,
        top_k=2,
    )

    print("\nResults:")

    for document, metadata, score in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["scores"],
    ):
        print("\n----------------------------")
        print(f"Score: {score:.4f}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Text: {document}")


if __name__ == "__main__":
    main()