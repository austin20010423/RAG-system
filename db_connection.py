import chromadb
import uuid
from readFile import get_chunk
from faiss_search import GetEmbeddedVector


def create_db(db_name: str, path: str):
    client = chromadb.PersistentClient(path=path)
    collection = client.create_collection(name=db_name)
    return collection


def connect_db(db_name: str, path: str):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(name=db_name)
    return collection


def add_data(collection, document: list, data=None):
    id_vector = [str(uuid.uuid4()) for i in range(1, len(chunk_array)+1)]
    collection.add(
        documents=document,
        ids=id_vector,
        metadatas=data
    )


def query_data(collection, text: str, num_results: int):

    return collection.query(
        query_texts=[text],
        n_results=num_results
    )

    chunk_array = get_chunk()

if __name__ == "__main__":

    chunk_array = get_chunk()
    collection = connect_db(db_name='chroma_LLM', path='chromadb')
    add_data(collection=collection, document=chunk_array)
    print(query_data(collection, text='外殼安裝', num_results=5))
