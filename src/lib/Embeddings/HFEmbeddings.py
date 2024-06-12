from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.lib.common.tools import get_device


class HFE5Embeddings(Embeddings):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2", model_kwargs={"device": get_device()}
        )

    def embed_documents(self, docs):
        embeddings = self.embedding_model.embed_documents(docs)
        return embeddings

    def embed_query(self, query):
        return self.embedding_model.embed_documents([query])[0]
    
    

class HFMultilingualE5Embeddings(Embeddings):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large", model_kwargs={"device": get_device()}
        )

    def embed_documents(self, docs):
        embeddings = self.embedding_model.embed_documents(docs)
        return embeddings
    
    def embed_query(self, query):
        return self.embedding_model.embed_documents([query])[0]
