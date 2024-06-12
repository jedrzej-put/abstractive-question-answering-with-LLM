import abc

class AbstractVectorDb:
    @abc.abstractmethod
    def load_vector_db(self):
        pass

    @abc.abstractmethod
    def store_embeddings(self, docs_embeddings_pairs, metadata):
        pass

    @abc.abstractmethod
    def similarity_search(self, query, n_docs=3):
        pass