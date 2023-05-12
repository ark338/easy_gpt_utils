#encoding=utf-8
import os
import sys
import openai
from tqdm import tqdm
import faiss
import numpy as np
import argparse
import pickle

# this python script is used to generate the embedding of the input file or folder

class Embedding():
    def __init__(self, model="text-embedding-ada-002", api_type="open_ai", api_base=None, api_key=None, api_version=None):
        if (api_type != "open_ai") and (api_type != "azure"):
            raise Exception("api_type should be open_ai or azure")

        self.model = model
        self.engine = model
        self.api_type = api_type
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
    
    # return embedding of the input text
    def __call__(self, text):
        return self.get_embedding([text]);

    # return embedding of the input text list
    def get_embedding(self, input_text_list):
        # if input_text_list is not a list throw an exception
        if not isinstance(input_text_list, list):
            raise TypeError("input_text_list should be a list")

        # openai api do not allow set model and engine at the same time
        if (self.api_type == "open_ai"):
            embedding = openai.Embedding.create(
                model=self.model, 
                input=input_text_list, 
                api_type=self.api_type, 
                api_key=self.api_key, 
                api_base=self.api_base, 
                api_version=self.api_version)
        elif (self.api_type == "azure"):
            embedding = openai.Embedding.create(
                input=input_text_list, 
                engine=self.engine,
                api_type=self.api_type, 
                api_key=self.api_key, 
                api_base=self.api_base, 
                api_version=self.api_version)
        return [(text, data.embedding) for text, data in zip(input_text_list, embedding.data)], embedding.usage.total_tokens

    def get_raw_embedding(self, raw_text: str):
         # openai api do not allow set model and engine at the same time
        if (self.api_type == "open_ai"):
            embedding = openai.Embedding.create(
                model=self.model, 
                input=raw_text,
                api_type=self.api_type, 
                api_key=self.api_key, 
                api_base=self.api_base, 
                api_version=self.api_version)
        elif (self.api_type == "azure"):
            embedding = openai.Embedding.create(
                input=raw_text,
                engine=self.engine,
                api_type=self.api_type, 
                api_key=self.api_key, 
                api_base=self.api_base, 
                api_version=self.api_version)
            
        return list(embedding.data[0].embedding)
    
    def create_embeddings(self, input_text_list):
        # if input_text_list is not a list throw an exception
        if not isinstance(input_text_list, list):
            raise TypeError("input_text_list should be a list")

        result = []
        lens = [len(text) for text in input_text_list]
        query_len = 0
        start_index = 0
        tokens = 0

        for index, l in tqdm(enumerate(lens)):
            query_len += l
            if query_len > 4096:
                ebd, tk = self.get_embedding(input_text_list[start_index:index + 1])
                query_len = 0
                start_index = index + 1
                tokens += tk
                result.extend(ebd)

        if query_len > 0:
            ebd, tk = self.get_embedding(input_text_list[start_index:])
            tokens += tk
            result.extend(ebd)
        return result, tokens

    def create_embeddings_from_text(self, text: str):
        # if text is not a string throw an exception
        if not isinstance(text, str):
            raise TypeError("text should be a string")

        return self.create_embeddings([text.strip() for text in text.splitlines() if text.strip()])

    def create_embedding_from_file(self, input_file, output_file= None):
        # if input_file is not a file throw an exception
        if not os.path.isfile(input_file):
            raise TypeError("input_file should be a file")

        with open(input_file, "r", encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
            embeddings, tokens = self.create_embeddings(texts)

        if output_file == None:
            return embeddings
        else:
            # pickle the embeddings
            with open(output_file, "wb") as f:
                pickle.dump(embeddings, f)

    def create_embedding_from_file_save_to_file(self, input_file, output_file):
        # if input_file is not a file throw an exception
        if not os.path.isfile(input_file):
            raise TypeError("input_file should be a file")

        with open(input_file, "r") as f:
            with open(output_file, "w") as of:
                of.write(self.create_embeddings(f.readlines()))

# unit tests
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=False, help="output file")
    parser.add_argument("--model", type=str, default="text-embedding-ada-002", help="model name")
    args = parser.parse_args()

    #embedding = Embedding(model=args.model)
    #print(embedding.create_embeddings(args.input))
    #print(embedding("hello world"))
    #print(embedding.create_embedding_from_file(args.input, args.output))
    #print(embedding.create_embedding_from_file_save_to_file(args.input, args.output))
    
    use_openai = True
    if use_openai:
        # test for openai
        embedding = Embedding()
    else :
        # test for azure
        embedding = Embedding(
            model="model-text-embedding-ada-002", 
            api_type="azure", 
            api_key = os.getenv("AZURE_API_KEY"),
            api_base = "https://ninebot-rd-openai-1.openai.azure.com/",
            api_version = "2022-12-01")
    
    print(embedding.create_embeddings_from_text(args.input))
    print(embedding.get_raw_embedding(args.input))
