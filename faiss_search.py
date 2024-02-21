from time import time
from transformers import (
    BertTokenizerFast,
    AutoModel,
)
import torch
import numpy as np
import faiss

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')


def GetEmbeddedVector(input_text: str) -> np.array:

    inputs = tokenizer(input_text, return_tensors="pt",
                       padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Retrieve the embedding vector for the [CLS] token
    embedding_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    # print("Vector Shape: ", embedding_vector.shape)
    return embedding_vector


def add_to_faiss_index(embeddings):
    vector = np.array(embeddings)
    # print(vector.shape)
    index = faiss.IndexFlatL2(vector.shape[1])
    index.add(vector)
    return index


def vector_search(index, query_embedding, text_array, k=5):
    distances, indices = index.search(
        np.array([query_embedding]), k)
    return [[text_array[i], float(dist)] for dist, i in zip(distances[0], indices[0])]


def data_to_faissindex(data_array):
    embedding_array = [GetEmbeddedVector(sentence) for sentence in data_array]
    faiss_index = add_to_faiss_index(embedding_array)
    return faiss_index
