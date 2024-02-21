from gpt4all import GPT4All
from faiss_search import *
from readFile import get_chunk
from time import time
import tkinter as tk
from tkinter import messagebox


def get_search_content(faiss_index, input_vector, data_array):
    start = time()
    search_result = vector_search(faiss_index, input_vector, data_array)
    end = time()
    print("Search Time: {:.3f} s".format(end-start))

    retrieve_text = [row[0] for row in search_result]
    return retrieve_text


def combine_data(retrieved_data: list, user_input: str):
    # system information
    system_prompt = """"你的角色：
                        你是一位樂於助人、受人尊敬且誠實的智能客服，永遠使用繁體中文和給予的資料回答問題。每一題都簡答即可。不需重複說明"""

    get_all = "\n".join(retrieved_data) + "\n\n"
    prompt = "資料庫資訊:\n" + get_all + "使用者" + user_input + system_prompt
    return prompt


def chat(prompt: str):

    model = GPT4All("gpt4all-falcon-newbpe-q4_0.gguf")
    start = time()
    response = model.generate(
        prompt, max_tokens=4096, streaming=True)

    try:
        while True:
            token = next(response)
            print(token, end='')
    except StopIteration:
        pass

    end = time()
    # print("Responses: ", response)
    print("\n\nResponse Time: {:.3f} s".format(end-start))


def main():

    data_array = get_chunk()

    # create faiss index
    embedding_array = [GetEmbeddedVector(sentence) for sentence in data_array]
    faiss_index = add_to_faiss_index(embedding_array)

    # user prompt
    user = str("藍芽dongle是否能偵測到server")
    input_vector = GetEmbeddedVector(user)

    # faiss search results
    search_result = get_search_content(faiss_index, input_vector, data_array)
    print("Search Result: ", search_result)


""""sumary_line

Keyword arguments:
argument -- description
Return: return_description
"""


main()
