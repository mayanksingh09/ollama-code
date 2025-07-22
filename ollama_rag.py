import ollama
import chromadb
from tqdm import tqdm
import argparse
import json

def generate_documents():
    documents = [
        "Huggingface is a beauty brand that specializes in skincare products.",
        "Ollama is the name of a battlestarship in the galaxy named Xenanao.",
        "ChromaDB is a mattress maker based out of Halifax, Nova Scotia."
    ]
    return documents


def get_embedding(text, model_name="nomic-embed-text"):

    embd_val = ollama.embed(model_name, text)
    return embd_val["embeddings"][0]


def generate_all_embeddings(documents, model_name="nomic-embed-text"):
    client = chromadb.Client()
    collection = client.create_collection(name="document_embeddings")
    for i, d in tqdm(enumerate(documents)):
        response = ollama.embed(model_name, d)
        embeddings = response["embeddings"][0]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d]
        )
    return collection


def retrieve_similar_documents(query, collection, model_name="nomic-embed-text", n_results=3):

    query_embedding = get_embedding(query, model_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

def rag_output(query, collection, generate_model, embed_model="nomic-embed-text", n_results=3):
    results = retrieve_similar_documents(query, collection, embed_model, n_results)
    output = []
    for doc in results["documents"]:
        output.append(doc)

    print("Retrieved documents: ", output)

    llm_response = ollama.generate(
        model=generate_model,
        prompt=f"Using this context: {output}, respond to this prompt: {query}"
    )
    return llm_response

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG with Ollama and ChromaDB")
    parser.add_argument("--generate_model", type=str, default="qwen2.5:1.5b", help="LLM model for generation")
    parser.add_argument("--embed_model", type=str, default="nomic-embed-text", help="Embedding model name")
    parser.add_argument("--n_results", type=int, default=1, help="Number of results to retrieve")
    parser.add_argument("--query", type=str, required=True, help="User Query")
    return parser.parse_args()

def main():

    args = parse_arguments()

    documents = generate_documents()
    collection = generate_all_embeddings(documents, args.embed_model)
    generated_output = rag_output(
        query=args.query, 
        collection=collection, 
        generate_model=args.generate_model, 
        embed_model=args.embed_model, 
        n_results=args.n_results)
    print(generated_output["response"])


if __name__ == "__main__":

    # documents = generate_documents()
    # print("Documents generated.")
    # collection = generate_all_embeddings(documents)
    # print("Embeddings generated.")
    #
    # generate_model = "qwen2.5:1.5b"
    # embed_model_name = "nomic-embed-text"
    #
    # query = "What is Ollama?"
    #
    # generated_output = rag_output(query, collection, generate_model, embed_model_name, n_results=1)
    # print("RAG output generated.")
    # print(generated_output["response"])   
    main()
