from langchain.vectorstores import FAISS
from third_parties import embeddings


def get_FAISS(docs):
    hf = embeddings.HuggingFace_embeddings()

    return FAISS.from_documents(documents=docs, embedding=hf)