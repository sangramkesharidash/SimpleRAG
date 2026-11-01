import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "./chroma_db_dynamic"

def inspect_and_query():
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Directory {CHROMA_PATH} does not exist.")
        return

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # 1. Get total stored chunk count
    collection = vectorstore._collection
    total_docs = collection.count()
    print(f"📊 Total Chunks in DB: {total_docs}\n" + "="*40)

    if total_docs == 0:
        print("Vector store is empty.")
        return

    # 2. Run a similarity search with relevance scores
    query = "apple-ddm CPU limits and memory requests"
    print(f"🔍 Running Test Query: '{query}'\n")

    # similarity_search_with_score returns (doc, distance_score)
    # Lower score = closer vector match in L2/Euclidean space
    results = vectorstore.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Result {i} (Distance Score: {score:.4f}) ---")
        print(f"Environment: {doc.metadata.get('environment', 'N/A')}")
        print(f"Source File: {doc.metadata.get('source_file', 'N/A')}")
        print(f"Content Snippet:\n{doc.page_content[:200]}...\n")

if __name__ == "__main__":
    inspect_and_query()
