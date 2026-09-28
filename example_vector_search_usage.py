from document_retrieval import retrieve_relevant_documents

if __name__ == "__main__":
    hits = retrieve_relevant_documents(
        "RELIANCE",
        "quarterly earnings telecom expansion",
        top_k=3,
    )
    for hit in hits:
        print(hit["title"], hit["source_url"], hit["score"])
