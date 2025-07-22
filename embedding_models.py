import ollama


if __name__ == "__main__":
    model_name = "nomic-embed-text"

    embd_val = ollama.embed(model_name, "Hello world")
    print(embd_val["embeddings"][0])  # Print the embeddings for the text "Hello world"
