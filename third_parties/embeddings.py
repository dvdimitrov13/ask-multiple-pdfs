from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

def HuggingFace_embeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )